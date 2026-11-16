import re

from .dashboard_models import (
    DashboardCampaignSetup,
    DashboardCompressedData,
    DashboardExperimentData,
    DashboardSingleSimulationData,
)
from .paths import RESULTS_DIR, RUN_VERSION
from .results import (
    ActiveLearningEngineeringSingleSimulationResult,
    ActiveLearningMultipleSimulationResult,
    ActiveLearningSingleSimulationResult,
)


def campaign_setup(ssr: ActiveLearningSingleSimulationResult) -> DashboardCampaignSetup:
    """Setup of one campaign. Screening campaigns draw from the pool without delta embeddings."""
    config = ssr.al_simulation_config
    stopping = config.stopping_config
    engineering = isinstance(ssr, ActiveLearningEngineeringSingleSimulationResult)
    generate_candidates = engineering and config.generate_candidates
    return DashboardCampaignSetup(
        n_start=len(config.start_ids) if config.start_ids else config.n_start,
        n_suggestions_per_iteration=config.n_suggestions_per_iteration,
        n_max_iterations=stopping.n_max_iterations,
        max_labels_budget=stopping.max_labels_budget,
        n_hits=stopping.n_hits,
        max_consecutive_failures=stopping.max_consecutive_failures,
        hit_percentile=config.hit_percentile,
        generate_candidates=generate_candidates,
        n_substitutions=config.n_substitutions if generate_candidates else None,
        use_delta_embeddings=engineering
        and ssr.al_campaign_config.use_delta_embeddings,
    )


def parse_dataset_from_filename(fname: str) -> str:
    # al_sim_{DATASET}_{EMBEDDER_NAME}_{MODEL_NAME}.json
    match = re.match(r"al_sim_(.+?)_(.+?)_(.+)\.json", fname)
    if match:
        return match.group(1)
    return "Unknown"


def compress_reports(run_name: str):
    if not RESULTS_DIR.exists():
        print(f"Directory {RESULTS_DIR} not found.")
        return

    json_files = sorted(RESULTS_DIR.glob("*.json"))
    experiments = []

    for path in json_files:
        if "compressed_dashboard" in path.name:
            continue

        print(f"Processing {path.name}...")

        multi = ActiveLearningMultipleSimulationResult.from_json(path)
        dataset_name = multi.dataset_id()
        embedder_name = multi.embedder_name()
        model_name = multi.model_type().value
        split_id = multi.split_id()
        potential_hits = multi.potential_hits()
        start_sets = [
            set(ssr.al_simulation_config.start_ids or [])
            for ssr in multi.simulation_results
        ]
        start_set_hits = sorted(set(potential_hits).intersection(*start_sets))

        # Prepare data for DashboardExperimentData
        summary = multi.summary()

        per_sim_n_hits = [
            list(map(len, ssr.simulation_result.iteration_hits or []))
            for ssr in multi.simulation_results
        ]
        per_sim_metrics_total = [
            [m.mean for m in ssr.simulation_result.iteration_metrics_total or []]
            for ssr in multi.simulation_results
        ]
        per_sim_metrics_suggestions = [
            [m.mean for m in ssr.simulation_result.iteration_metrics_suggestions or []]
            for ssr in multi.simulation_results
        ]
        per_sim_is_success = [ssr.is_success() for ssr in multi.simulation_results]

        # Prepare single_sims for drill-down
        single_sims = []
        for ssr in multi.simulation_results:
            # We only need enough to render the single view
            # Most of it is in ssr.simulation_result and ssr.al_campaign_config
            # We can store a stripped down version
            sim_data = DashboardSingleSimulationData(
                label=ssr.label(),
                is_success=ssr.is_success(),
                seed=ssr.al_campaign_config.seed,
                stop_reasons=ssr.simulation_result.stop_reasons or ["None"],
                iteration_metrics_total=[
                    m.mean for m in ssr.simulation_result.iteration_metrics_total or []
                ],
                iteration_metrics_suggestions=[
                    m.mean
                    for m in ssr.simulation_result.iteration_metrics_suggestions or []
                ],
                iteration_hits=list(ssr.simulation_result.iteration_hits or []),
                iteration_consecutive_failures=list(
                    ssr.simulation_result.iteration_consecutive_failures or []
                ),
                iteration_results_count=len(
                    ssr.simulation_result.iteration_results or []
                ),
                n_hits_threshold=ssr.al_simulation_config.stopping_config.n_hits,
            )
            single_sims.append(sim_data)

        exp_data = DashboardExperimentData(
            name=path.name,
            dataset_id=dataset_name,
            embedder=embedder_name,
            model=model_name,
            split_id=split_id,
            setup=campaign_setup(multi.simulation_results[0]),
            summary=summary,
            aggregated_hits=multi.get_aggregated_hits(),
            aggregated_suggestions=multi.get_aggregated_number_of_suggestions(),
            potential_hits=potential_hits,
            start_set_hits=start_set_hits,
            per_sim_n_hits=per_sim_n_hits,
            per_sim_metrics_total=per_sim_metrics_total,
            per_sim_metrics_suggestions=per_sim_metrics_suggestions,
            per_sim_is_success=per_sim_is_success,
            single_sims=single_sims,
        )
        experiments.append(exp_data)

    compressed = DashboardCompressedData(run_name=run_name, experiments=experiments)
    OUTPUT_FILE = RESULTS_DIR / f"compressed_dashboard_data_{run_name}.json"

    with open(OUTPUT_FILE, "w") as f:
        f.write(compressed.model_dump_json(indent=4))

    print(f"Successfully created {OUTPUT_FILE} with {len(experiments)} experiments.")


if __name__ == "__main__":
    compress_reports(RUN_VERSION)
