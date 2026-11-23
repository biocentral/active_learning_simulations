"""Runs engineering experiments

TODO if this blows up more, use a proper configuration management tool for this

Throughput regimes:
  - low   10 starting variants, 8 rounds of 5 measured variants each  ->  40 measured
  - high  96 starting variants, 8 rounds of 48 measured variants each -> 384 measured

Starting variants always contain WT plus randomly drawn variants

Mutation generation per default is random sampling from the annotated landscape.
--candidate-source uses proper mutation generation (restricted to annotated variants).

Result JSON filenames are named by throughput tag + CampaignSettings fingerprint
"""

import json
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import timedelta
from itertools import product
from pathlib import Path
from queue import Queue
from time import monotonic
from urllib.request import urlopen

from biocentral_api import ActiveLearningModelType, CommonEmbedder
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from biocentral_al_sim.datasets import ALSimulatorDataset
from biocentral_al_sim.paths import RESULTS_DIR
from biocentral_al_sim.results import (
    ActiveLearningMultipleSimulationResult,
)
from biocentral_al_sim.settings import (
    THROUGHPUTS,
    ActiveLearningFixedBaseConfig,
    CampaignSettings,
    EngineeringCampaignSettings,
)
from biocentral_al_sim.simulators import (
    ActiveLearningEngineeringSimulator,
    biocentral_api,
    profile_summary,
)
from biocentral_al_sim.splits import (
    AllOf,
    ALSimulatorSplitDefinition,
    AnyOf,
    MutationCount,
    RandomSample,
    SelectorSplit,
)

DATASET_CHOICES = {
    "SACAS9": ALSimulatorDataset.COMBINGYM_SACAS9,  #   1_296 variants, MAXIMIZE
    "MTAGBFP2": ALSimulatorDataset.COMBINGYM_MTAGBFP2,  #   8_192 variants, MAXIMIZE
    "CR9114": ALSimulatorDataset.COMBINGYM_CR9114,  #  65_094 variants, MAXIMIZE
    "GB1": ALSimulatorDataset.COMBINGYM_GB1,  # 149_361 variants, MAXIMIZE
    "CREILOV": ALSimulatorDataset.COMBINGYM_CREILOV,  # 165_428 variants, MAXIMIZE
    "TRPB": ALSimulatorDataset.FLIP2_TRPB,  # 159_129 variants once restricted
    "NUCB": ALSimulatorDataset.FLIP2_NUCB,  #   2_088 variants once restricted
}

EMBEDDERS = [
    CommonEmbedder.ONE_HOT_ENCODING.value,
    CommonEmbedder.LENGTH_EMBEDDER.value,
    CommonEmbedder.RANDOM_EMBEDDER.value,
    CommonEmbedder.BLOSUM62.value,
    CommonEmbedder.ESM2_8M.value,
    CommonEmbedder.ESM2_650M.value,
    CommonEmbedder.ProtT5.value,
]
# These don't make sense with delta embeddings
# length & random embedder don't have residue level
# one-hot & BLOSUM62 get constant shift without any residue information
BASELINE_EMBEDDERS = {
    CommonEmbedder.ONE_HOT_ENCODING.value,
    CommonEmbedder.LENGTH_EMBEDDER.value,
    CommonEmbedder.RANDOM_EMBEDDER.value,
    CommonEmbedder.BLOSUM62.value,
}
MODEL_TYPES = [
    ActiveLearningModelType.RANDOM,
    ActiveLearningModelType.GAUSSIAN_PROCESS,
    ActiveLearningModelType.FNN_MCD,
]
# Only needs to run once
RANDOM_BASELINE_EMBEDDER = CommonEmbedder.ONE_HOT_ENCODING.value


def engineering_settings(
    settings: CampaignSettings,
    use_delta_embeddings: bool,
    generate_candidates: bool,
    n_substitutions: int,
) -> EngineeringCampaignSettings:
    return EngineeringCampaignSettings(
        **settings.model_dump(),
        use_delta_embeddings=use_delta_embeddings,
        generate_candidates=generate_candidates,
        n_substitutions=n_substitutions,
    )


def starting_set(
    dataset_id: ALSimulatorDataset, pool, n_start: int, seed: int
) -> list[str]:
    """Wildtype plus `n_start - 1` variants drawn from the whole pool."""
    definition = ALSimulatorSplitDefinition(
        description=f"Wild type plus {n_start - 1} variants drawn from the whole pool.",
        strategy=SelectorSplit(
            selector=AnyOf(
                selectors=[
                    MutationCount(max_mutations=0),
                    AllOf(
                        selectors=[
                            MutationCount(min_mutations=1),
                            RandomSample(n=n_start - 1, seed=seed),
                        ]
                    ),
                ]
            )
        ),
    )
    _, assignment = definition.resolve(
        pool, explicit_reference=dataset_id.reference_sequence()
    )
    return assignment.train_ids


def result_file_name(
    dataset_id: ALSimulatorDataset,
    embedder_name: str,
    model_type: ActiveLearningModelType,
    settings: EngineeringCampaignSettings,
) -> str:
    """Name of an experiment's result file."""
    delta = "_delta" if settings.use_delta_embeddings else ""
    # The fingerprint already separates the arms; the tag is so a directory listing reads
    generated = (
        f"_gen{settings.n_substitutions}" if settings.generate_candidates else ""
    )
    return (
        f"al_eng_{dataset_id.name}_{embedder_name.replace('/', '-')}"
        f"_{model_type.value}_{settings.short_name}{delta}{generated}"
        f"_{settings.fingerprint()}.json"
    )


def run_experiment(
    dataset_id: ALSimulatorDataset,
    embedder_name: str,
    model_type: ActiveLearningModelType,
    settings: EngineeringCampaignSettings,
    show_progress: bool = True,
    on_replicate: Callable[[int], None] | None = None,
    commits: dict[str, str | None] | None = None,
) -> ActiveLearningMultipleSimulationResult:
    """Run one experiment over `settings.n_replicates` replicates, skipping those already saved.

    Args:
        show_progress: Print each replicate and show the client's progress bar.
        on_replicate: Called with the replicate index as each replicate starts.
        commits: Recorded on each replicate this call runs.

    Returns:
        The first `settings.n_replicates` replicates' results, saved under `result_file_name`.
    """
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    save_path = Path(RESULTS_DIR) / result_file_name(
        dataset_id, embedder_name, model_type, settings
    )
    # Replicate i always runs on seed first_seed + i, so a saved file holds a prefix of them
    done = (
        ActiveLearningMultipleSimulationResult.from_json(save_path).simulation_results
        if save_path.exists()
        else []
    )
    if len(done) >= settings.n_replicates:
        return ActiveLearningMultipleSimulationResult(done[: settings.n_replicates])

    pool = dataset_id.pool()
    definition = dataset_id.definition()
    results = []
    for replicate in range(len(done), settings.n_replicates):
        seed = settings.first_seed + replicate
        draw_seed = seed if settings.redraw_start_set else settings.first_seed
        base_config = ActiveLearningFixedBaseConfig(
            dataset_id=dataset_id,
            simulation_data=pool,
            settings=settings,
            start_ids=starting_set(dataset_id, pool, settings.n_start, draw_seed),
            optimization_mode=definition.optimization_mode,
            target_lb=definition.target_lb,
            target_ub=definition.target_ub,
            target_value=definition.target_value,
            discrete_targets=definition.discrete_targets,
        )
        if on_replicate is not None:
            on_replicate(replicate)
        if show_progress:
            print(
                f"  replicate {replicate + 1}/{settings.n_replicates} "
                f"(campaign seed {seed}, draw seed {draw_seed})",
                flush=True,
            )
        result = ActiveLearningEngineeringSimulator(base_config).run_replicate(
            model_type=model_type,
            embedder_name=embedder_name,
            seed=seed,
            show_progress=show_progress,
        )
        result.commits = commits
        results.append(result)

    multi = ActiveLearningMultipleSimulationResult(done + results)
    multi.save(save_path)
    if show_progress:
        multi.print_stats()
    return multi


def git_commit(repo: Path) -> str:
    """Full hash of `repo`'s HEAD, suffixed `-dirty` for uncommitted changes."""
    describe = ["describe", "--always", "--dirty", "--abbrev=40", "--exclude=*"]
    return subprocess.run(
        ["git", "-C", str(repo), *describe], capture_output=True, text=True, check=True
    ).stdout.strip()


SHORT_EMBEDDER = {
    "one_hot_encoding": "onehot",
    "length_embedder": "length",
    "random_embedder": "random",
    "blosum62": "blosum",
    "facebook/esm2_t6_8M_UR50D": "esm2-8M",
    "facebook/esm2_t33_650M_UR50D": "esm2-650M",
    "Rostlab/prot_t5_xl_uniref50": "protT5",
}


Combination = tuple[
    EngineeringCampaignSettings, ALSimulatorDataset, str, ActiveLearningModelType
]


def _label(combination: Combination) -> str:
    """Progress row label in rich markup: dataset, embedder, model type and throughput."""
    settings, dataset_id, embedder_name, model_type = combination
    return (
        f"[cyan]{dataset_id.name.replace('COMBINGYM_', '')}[/]"
        f" [magenta]{SHORT_EMBEDDER.get(embedder_name, embedder_name)}[/]"
        f" [yellow]{model_type.value}[/] [dim]{settings.short_name}[/]"
    )


@contextmanager
def _borrowed(slots: "Queue[int]"):
    """Hand a display row to a worker thread for the duration of one experiment."""
    slot = slots.get()
    try:
        yield slot
    finally:
        slots.put(slot)


class StepElapsedColumn(ProgressColumn):
    """Time spent on the current step.

    `TimeElapsedColumn` measures the whole task, which for a grid row spans every
    replicate. This one restarts whenever `step_started` is refreshed.
    """

    def render(self, task) -> Text:
        started = task.fields.get("step_started")
        if started is None:
            return Text("")
        return Text(str(timedelta(seconds=int(monotonic() - started))), style="cyan")


def _progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.fields[step]}", style="dim"),
        StepElapsedColumn(),
        SpinnerColumn(style="cyan"),
        BarColumn(
            bar_width=None,
            style="grey30",
            complete_style="green",
            finished_style="bright_green",
        ),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )


def main(
    settings: list[EngineeringCampaignSettings],
    datasets: list[ALSimulatorDataset],
    jobs: int = 1,
) -> int:
    """Run each campaign setting on every dataset, embedder and model type, skipping saved results.

    Args:
        settings: Campaign settings to run.
        datasets: Datasets to run.
        jobs: Experiments run concurrently.

    Returns:
        Exit code: 0 if every experiment succeeded, 1 otherwise.
    """
    combinations = [
        (settings, dataset_id, embedder_name, model_type)
        for settings in settings
        for dataset_id in datasets
        for embedder_name in EMBEDDERS
        for model_type in MODEL_TYPES
        # Delta embeddings add nothing for the baseline embedders
        if not (settings.use_delta_embeddings and embedder_name in BASELINE_EMBEDDERS)
        and (
            model_type != ActiveLearningModelType.RANDOM
            or (
                embedder_name == RANDOM_BASELINE_EMBEDDER
                and not settings.use_delta_embeddings
            )
        )
    ]
    total = len(combinations)
    failures: list[tuple[str, str]] = []

    def name_of(combination: Combination) -> str:
        settings, dataset_id, embedder_name, model_type = combination
        return result_file_name(dataset_id, embedder_name, model_type, settings)

    # The health check also warms the cached client before any worker starts
    server_url = next(
        url for url, healthy, _ in biocentral_api().get_health_status() if healthy
    )
    with urlopen(f"{server_url}/health") as response:
        server_commit = json.load(response).get("commit")
    commits = {
        "active_learning": git_commit(Path(__file__).parent),
        "server": server_commit,
    }
    print(f"Commits: {commits}")

    with _progress() as progress:
        regimes = ", ".join(dict.fromkeys(settings.short_name for settings in settings))
        overall = progress.add_task(
            f"[bold]grid[/] [dim]{regimes}[/]", total=total, step="", step_started=None
        )
        # Create one row row per concurrent job
        # from the start so display height doesn't change
        rows = [
            progress.add_task(
                "idle", total=1, visible=False, step="", step_started=None
            )
            for _ in range(jobs)
        ]
        slots: Queue[int] = Queue()
        for index in range(jobs):
            slots.put(index)

        def run_one(combination: Combination) -> None:
            settings = combination[0]
            with _borrowed(slots) as slot:
                row, label = rows[slot], _label(combination)
                progress.reset(
                    row,
                    total=settings.n_replicates,
                    description=label,
                    visible=True,
                    step="",
                    step_started=None,
                )
                try:
                    run_experiment(
                        *combination[1:],
                        settings,
                        # The client's own bar cannot share a terminal with this display.
                        show_progress=False,
                        commits=commits,
                        on_replicate=lambda index: progress.update(
                            row,
                            completed=index,
                            step=f"replicate {index + 1}/{settings.n_replicates}",
                            step_started=monotonic(),
                        ),
                    )
                    progress.update(row, completed=settings.n_replicates)
                finally:
                    progress.update(row, visible=False)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(run_one, c): c for c in combinations}
            for future in as_completed(futures):
                name = name_of(futures[future])
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001
                    failures.append((name, repr(error)))
                    progress.console.print(f"[red]FAILED[/] {name}: {error!r}")
                progress.advance(overall)

    done = total - len(failures)
    print("\n" + profile_summary())
    print(f"\nGrid finished: {done}/{total} succeeded.")
    for name, error in failures:
        print(f"  FAILED {name}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--throughput", nargs="+", choices=[*THROUGHPUTS], default=[*THROUGHPUTS]
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[*DATASET_CHOICES],
        required=True,
        metavar="NAME",
        help=f"Datasets to run. Choices: {', '.join(DATASET_CHOICES)}",
    )
    parser.add_argument(
        "--fixed-start-set",
        action="store_true",
        help="Share one starting set across replicates instead of redrawing",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=None,
        metavar="N",
        help="Seeded repeats per experiment; default 5. Raising it on a saved "
        "experiment runs only the missing replicates.",
    )
    parser.add_argument(
        "--delta-embeddings",
        nargs="+",
        choices=["off", "on"],
        default=["off"],
        help="Subtract the wildtype embedding from each variant embedding. "
        "Skipped for baseline embedders, which the server rejects.",
    )
    parser.add_argument(
        "--candidate-source",
        nargs="+",
        choices=["pool", "generated"],
        default=["pool"],
        help="How to generate candidates: 'pool' draws from the landscape at random, "
        "'generated' mutates variants measured already.",
    )
    parser.add_argument(
        "--n-substitutions",
        type=int,
        default=1,
        metavar="K",
        help="Number of substitutions to add to measured sequences each round",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Run N experiments concurrently. Needs enough biocentral-worker "
        "replicas to avoid waiting and SERVER_DEBUG set on the server "
        "to avoid throttling",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.n_substitutions < 1:
        parser.error("--n-substitutions must be >= 1")
    overrides: dict[str, object] = {}
    if args.fixed_start_set:
        overrides["redraw_start_set"] = False
    if args.replicates is not None:
        overrides["n_replicates"] = args.replicates
    deltas = [value == "on" for value in args.delta_embeddings]
    sources = [value == "generated" for value in args.candidate_source]
    variants = [
        engineering_settings(
            THROUGHPUTS[tag], delta, generated, args.n_substitutions
        ).model_copy(update=overrides)
        for tag, delta, generated in product(args.throughput, deltas, sources)
    ]
    chosen = [DATASET_CHOICES[name] for name in args.datasets]
    raise SystemExit(main(variants, datasets=chosen, jobs=args.jobs))
