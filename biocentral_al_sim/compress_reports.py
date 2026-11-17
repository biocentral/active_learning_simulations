import re
from statistics import NormalDist

import numpy as np
import pandas as pd
from biocentral_api import ActiveLearningOptimizationMode
from biotrainer_core.input_files import read_FASTA

from .dashboard_models import (
    UNCERTAINTY_CALIBRATION_LEVELS,
    DashboardCampaignSetup,
    DashboardCompressedData,
    DashboardExperimentData,
    DashboardSingleSimulationData,
)
from .datasets import ALSimulatorDataset
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


def dataset_labels(dataset_id: ALSimulatorDataset) -> dict[str, str]:
    """Labels of a dataset's variants by seq_id."""
    return {
        data_point.seq_id: data_point.label
        for data_point in read_FASTA(dataset_id.to_path())
        if data_point.label is not None
    }


def check_replicates(
    multi: ActiveLearningMultipleSimulationResult, labels: dict[str, str], name: str
) -> None:
    """Raise if a replicate doesn't have full results for each round or has variants with unknown IDs."""
    for index, ssr in enumerate(multi.simulation_results):
        result = ssr.simulation_result
        rounds = result.iteration_results or []
        if len(rounds) < len(result.iteration_hits or []):
            raise ValueError(f"{name}: replicate {index} lost a round's results")
        unknown = {
            seq_id
            for iteration in rounds
            for seq_id in iteration.suggestions
            if seq_id not in labels
        }
        if unknown:
            raise ValueError(
                f"{name}: replicate {index} holds unknown variant ids, e.g. {min(unknown)}"
            )


def best_suggested_labels(
    multi: ActiveLearningMultipleSimulationResult, labels: dict[str, str]
) -> list[float] | None:
    """Best label among each replicate's suggestions, None for discrete targets."""
    match multi.simulation_results[0].al_campaign_config.optimization_mode:
        case ActiveLearningOptimizationMode.MAXIMIZE:
            best = max
        case ActiveLearningOptimizationMode.MINIMIZE:
            best = min
        case _:
            return None
    return [
        best(
            float(labels[seq_id])
            for iteration in ssr.simulation_result.iteration_results or []
            for seq_id in iteration.suggestions
        )
        for ssr in multi.simulation_results
    ]


def _numeric_targets(multi: ActiveLearningMultipleSimulationResult) -> bool:
    return multi.simulation_results[0].al_campaign_config.optimization_mode in (
        ActiveLearningOptimizationMode.MAXIMIZE,
        ActiveLearningOptimizationMode.MINIMIZE,
    )


def suggested_label_means(
    multi: ActiveLearningMultipleSimulationResult, labels: dict[str, str]
) -> list[list[float]] | None:
    """Mean label of each round's suggestions per replicate, None for discrete targets."""
    if not _numeric_targets(multi):
        return None
    return [
        [
            float(np.mean([float(labels[seq_id]) for seq_id in iteration.suggestions]))
            for iteration in ssr.simulation_result.iteration_results or []
        ]
        for ssr in multi.simulation_results
    ]


def calibration_coverage(
    multi: ActiveLearningMultipleSimulationResult, labels: dict[str, str]
) -> list[list[float]] | None:
    """For each replicate: the share of scored candidates whose label lies within the central
    interval at each of UNCERTAINTY_CALIBRATION_LEVELS. None for discrete targets.

    Each prediction's uncertainty is the standard deviation of a normal distribution.
    """
    if not _numeric_targets(multi):
        return None
    z = np.array(
        [
            NormalDist().inv_cdf((1 + level) / 2)
            for level in UNCERTAINTY_CALIBRATION_LEVELS
        ]
    )
    coverage = []
    for ssr in multi.simulation_results:
        prediction, sigma, label = np.array(
            [
                (
                    float(result.prediction),
                    result.uncertainty,
                    float(labels[result.entity_id]),
                )
                for iteration in ssr.simulation_result.iteration_results or []
                for result in iteration.results
            ]
        ).T
        with np.errstate(divide="ignore", invalid="ignore"):
            normalised = np.abs(label - prediction) / sigma
        coverage.append([float(np.mean(normalised <= threshold)) for threshold in z])
    return coverage


NDCG_CUTOFFS = (10, 96)
NDCG_PERCENTAGES = (1, 10)


def percentage_cutoff(n_scored: int, percentage: int) -> int:
    return max(1, n_scored * percentage // 100)


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman correlation over average ranks, None when either side is constant."""
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def _ndcg(prediction: np.ndarray, label: np.ndarray, k: int) -> float | None:
    """NDCG@k of ranking by prediction, with the labels min-max scaled to [0, 1] as gains.
    None when every label is the same."""
    span = np.ptp(label)
    if span == 0:
        return None
    gains = (label - label.min()) / span
    top = np.argsort(-prediction, kind="stable")[:k]
    discounts = 1 / np.log2(np.arange(2, len(top) + 2))
    ideal = np.sort(gains)[::-1][: len(top)]
    return float((gains[top] * discounts).sum() / (ideal * discounts).sum())


def final_round_metrics(
    multi: ActiveLearningMultipleSimulationResult, labels: dict[str, str]
) -> list[dict[str, float | None]] | None:
    """Each replicate's MAE, RMSE, Spearman correlation and NDCG of the final round's predictions on
    the variants scored in that round. The ranking metrics are None when the predictions take at most two values.
    None for discrete targets."""
    if not _numeric_targets(multi):
        return None
    mode = multi.simulation_results[0].al_campaign_config.optimization_mode
    # Rank so that the best label comes first
    sign = 1.0 if mode == ActiveLearningOptimizationMode.MAXIMIZE else -1.0
    metrics = []
    for ssr in multi.simulation_results:
        final = (ssr.simulation_result.iteration_results or [])[-1]
        prediction = np.array([float(result.prediction) for result in final.results])
        label = np.array([float(labels[result.entity_id]) for result in final.results])
        # Two distinct values cannot rank the variants. The GP predicts so with the length
        # embedder, whose input is the same for every variant of a substitution landscape
        ranks = np.unique(prediction).size > 2
        cutoffs = {f"ndcg@{k}": k for k in NDCG_CUTOFFS} | {
            f"ndcg@{share}%": percentage_cutoff(len(label), share)
            for share in NDCG_PERCENTAGES
        }
        replicate = {
            "mae": float(np.mean(np.abs(prediction - label))),
            "rmse": float(np.sqrt(np.mean((prediction - label) ** 2))),
            "spearman": _spearman(prediction, label) if ranks else None,
        }
        for name, k in cutoffs.items():
            replicate[name] = (
                _ndcg(sign * prediction, sign * label, k) if ranks else None
            )
        metrics.append(replicate)
    return metrics


def round_errors(
    multi: ActiveLearningMultipleSimulationResult, labels: dict[str, str]
) -> dict[str, list[list[float]]] | None:
    """Per replicate and round, the MAE and RMSE of the round's predictions on its suggestions
    and on every candidate it scored. None for discrete targets."""
    if not _numeric_targets(multi):
        return None
    errors = {
        f"{metric}_{subset}": []
        for metric in ("mae", "rmse")
        for subset in ("suggestions", "scored")
    }
    for ssr in multi.simulation_results:
        replicate = {key: [] for key in errors}
        for iteration in ssr.simulation_result.iteration_results or []:
            predicted = {
                result.entity_id: float(result.prediction)
                for result in iteration.results
            }
            for subset, ids in (
                ("suggestions", iteration.suggestions),
                ("scored", list(predicted)),
            ):
                residual = np.array([predicted[i] - float(labels[i]) for i in ids])
                replicate[f"mae_{subset}"].append(float(np.mean(np.abs(residual))))
                replicate[f"rmse_{subset}"].append(float(np.sqrt(np.mean(residual**2))))
        for key, values in replicate.items():
            errors[key].append(values)
    return errors


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
    # Result files carry no labels, so suggestions are looked up in each dataset's FASTA
    labels_by_dataset = {}

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
        if dataset_name not in labels_by_dataset:
            labels_by_dataset[dataset_name] = dataset_labels(dataset_name)
        check_replicates(multi, labels_by_dataset[dataset_name], path.name)

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
            per_sim_n_suggestions=[
                ssr.get_total_number_of_suggestions()
                for ssr in multi.simulation_results
            ],
            per_sim_best_label=best_suggested_labels(
                multi, labels_by_dataset[dataset_name]
            ),
            per_sim_suggested_label_means=suggested_label_means(
                multi, labels_by_dataset[dataset_name]
            ),
            per_sim_calibration=calibration_coverage(
                multi, labels_by_dataset[dataset_name]
            ),
            per_sim_final_round_metrics=final_round_metrics(
                multi, labels_by_dataset[dataset_name]
            ),
            per_sim_round_errors=round_errors(multi, labels_by_dataset[dataset_name]),
            per_sim_final_round_scored=[
                len((ssr.simulation_result.iteration_results or [])[-1].results)
                for ssr in multi.simulation_results
            ],
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
