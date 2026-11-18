"""Streamlit dashboard for active learning simulation results.

Compress the results, then run the dashboard:
    uv run python -m biocentral_al_sim.compress_reports
    uv run streamlit run al_sim_dashboard.py
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from biocentral_api import CommonEmbedder, ProjectionResult
from biocentral_vis import BiocentralChart
from biotrainer_core.data_classes import SequenceData

from biocentral_al_sim import plots as al_plots
from biocentral_al_sim.compress_reports import percentage_cutoff
from biocentral_al_sim.dashboard_models import (
    UNCERTAINTY_CALIBRATION_LEVELS,
    DashboardCampaignSetup,
    DashboardCompressedData,
    DashboardExperimentData,
)
from biocentral_al_sim.datasets import ALSimulatorDataset
from biocentral_al_sim.paths import PROJECTIONS_DIR, RESULTS_DIR
from biocentral_al_sim.settings import THROUGHPUTS
from biocentral_al_sim.splits import substitutions

st.set_page_config(page_title="AL Simulation Dashboard", layout="wide")


ENCODER_LABELS = {
    CommonEmbedder.RANDOM_EMBEDDER.value: "random",
    CommonEmbedder.LENGTH_EMBEDDER.value: "length",
    CommonEmbedder.ONE_HOT_ENCODING.value: "OHE",
    CommonEmbedder.BLOSUM62.value: "Blosum62",
    CommonEmbedder.ESM2_8M.value: "ESM-2 (8M)",
    CommonEmbedder.ESM2_650M.value: "ESM-2 (650M)",
    CommonEmbedder.ProtT5.value: "ProtT5",
}
ENCODER_ORDER = list(ENCODER_LABELS.values())


def _get_encoder_label(embedder: str) -> str:
    return ENCODER_LABELS.get(embedder, embedder)


@st.cache_data(show_spinner=True)
def _get_run_files():
    return list(RESULTS_DIR.glob("compressed_dashboard_data_*.json"))


@st.cache_resource(show_spinner=True)
def _load_pool(selected_dataset: str) -> list[SequenceData]:
    """Sequences the dataset's campaigns drew from."""
    return ALSimulatorDataset[selected_dataset].pool()


@st.cache_data(show_spinner=True)
def _load_depths(selected_dataset: str) -> dict[str, int] | None:
    """Get number of mutations of each sequence. Returns None if there is no parent sequence."""
    reference = ALSimulatorDataset[selected_dataset].reference_sequence()
    if reference is None:
        return None
    return {
        data_point.seq_id: len(substitutions(data_point.seq, reference))
        for data_point in _load_pool(selected_dataset)
    }


@st.cache_data(show_spinner=False)
def _load_compressed(run_file_paths: list[Path]) -> dict[str, DashboardCompressedData]:
    result = {}  # name to compressed data
    for path in run_file_paths:
        try:
            compressed_data = DashboardCompressedData.model_validate_json(
                path.read_text()
            )
            result[compressed_data.run_name] = compressed_data
        except (OSError, ValueError) as exc:
            st.sidebar.error(f"Failed to load compressed data: {exc}")
            continue
    return result


@st.cache_data(show_spinner=False)
def _load_projection(
    dataset_id: ALSimulatorDataset, embedder_name: str
) -> ProjectionResult | None:
    embedder_name = embedder_name.replace("/", "-")
    projection_path = (
        PROJECTIONS_DIR / f"projection_result_{dataset_id.name}_{embedder_name}.json"
    )
    if projection_path.exists():
        return ProjectionResult.model_validate_json(projection_path.read_text())
    return None


def _describe_hits(exp: DashboardExperimentData) -> str:
    match exp.summary["optimization_mode"]:
        case "MAXIMIZE":
            return f"Pool members in the top {exp.setup.hit_percentile:g}% of the pool's labels."
        case "MINIMIZE":
            return f"Pool members in the bottom {exp.setup.hit_percentile:g}% of the pool's labels."
        case _:
            return "Pool members that meet the target."


def _compute_hit_threshold(
    exp: DashboardExperimentData, labels: list[float]
) -> float | None:
    """Compute hit threshold of annotations from servec's percentile rule."""
    match exp.summary["optimization_mode"]:
        case "MAXIMIZE":
            return float(np.percentile(labels, 100 - exp.setup.hit_percentile))
        case "MINIMIZE":
            return float(np.percentile(labels, exp.setup.hit_percentile))
        case _:
            return None


def _summarize_potential_hits(exp: DashboardExperimentData, n_pool: int) -> str:
    n_potential_hits = len(exp.potential_hits)
    return f"{n_potential_hits} ({100 * n_potential_hits / n_pool:.2f} % of the pool)"


def _describe_candidate_source(setup: DashboardCampaignSetup) -> str:
    if not setup.generate_candidates:
        return "Pool"
    plural = "" if setup.n_substitutions == 1 else "s"
    return f"Generated ({setup.n_substitutions} substitution{plural})"


def _describe_stopping_criteria(setup: DashboardCampaignSetup) -> str:
    criteria = []
    if setup.max_labels_budget is not None:
        criteria.append(f"a budget of {setup.max_labels_budget} labels")
    if setup.n_max_iterations is not None:
        criteria.append(f"{setup.n_max_iterations} rounds")
    if setup.n_hits is not None:
        criteria.append(f"{setup.n_hits} hits")
    if setup.max_consecutive_failures is not None:
        criteria.append(f"{setup.max_consecutive_failures} rounds without a hit")
    return "Stops at whichever comes first: " + ", ".join(criteria)


def _describe_throughput(setup: DashboardCampaignSetup) -> str:
    """Shape of the campaign, led by the name of the throughput preset it follows."""
    shape = (
        f"{setup.n_start} + {setup.n_max_iterations} × "
        f"{setup.n_suggestions_per_iteration}"
    )
    for preset in THROUGHPUTS.values():
        if (
            preset.n_start,
            preset.n_iterations,
            preset.n_suggestions_per_iteration,
        ) == (
            setup.n_start,
            setup.n_max_iterations,
            setup.n_suggestions_per_iteration,
        ):
            return f"{preset.short_name} ({shape})"
    return shape


def _get_arm(exp: DashboardExperimentData) -> tuple:
    return exp.dataset_id, exp.embedder, exp.model


def _filter_experiments(
    experiments: list[DashboardExperimentData],
) -> list[DashboardExperimentData]:
    """Filter a dataset's experiments to one throughput, candidate source and embedding
    variant."""
    st.sidebar.subheader("Campaign Setup")

    by_size = sorted(
        experiments,
        key=lambda e: (e.setup.n_start, e.setup.n_suggestions_per_iteration),
    )
    throughput = st.sidebar.radio(
        "Throughput",
        list(dict.fromkeys(_describe_throughput(e.setup) for e in by_size)),
        help="Starting set + rounds × sequences chosen per round",
    )
    experiments = [
        e for e in experiments if _describe_throughput(e.setup) == throughput
    ]

    by_source = sorted(experiments, key=lambda e: e.setup.n_substitutions or 0)
    source = st.sidebar.radio(
        "Candidate generation",
        list(dict.fromkeys(_describe_candidate_source(e.setup) for e in by_source)),
        help="Pool: random unseen variants of the landscape. Generated: mutants of the "
        "variants measured so far.",
    )
    experiments = [
        e for e in experiments if _describe_candidate_source(e.setup) == source
    ]

    if not any(e.setup.use_delta_embeddings for e in experiments):
        return experiments
    delta = st.sidebar.radio(
        "Delta embeddings",
        ["Off", "On"],
        horizontal=True,
    )
    if delta == "Off":
        return [e for e in experiments if not e.setup.use_delta_embeddings]
    with_delta = {_get_arm(e) for e in experiments if e.setup.use_delta_embeddings}
    return [
        e
        for e in experiments
        if e.setup.use_delta_embeddings or _get_arm(e) not in with_delta
    ]


def _index_by_label(
    exps: list[DashboardExperimentData],
) -> dict[str, DashboardExperimentData]:
    """Experiments by model and embedder. Ambiguous labels fall back to the file name."""
    exps = sorted(exps, key=lambda e: (e.model, e.embedder))
    labels = [
        f"{e.model} · {_get_encoder_label(e.embedder)}"
        + (" · delta" if e.setup.use_delta_embeddings else "")
        for e in exps
    ]
    counts = Counter(labels)
    return {
        label if counts[label] == 1 else e.name: e for e, label in zip(exps, labels)
    }


def _render_simulations_aggregate(exp: DashboardExperimentData) -> None:
    st.subheader("Overall performance summary")

    summary = exp.summary
    threshold = summary["n_hits_threshold"]
    cols = st.columns(3 if threshold is None else 4)
    cols[0].metric("Total number of hits", exp.aggregated_hits)
    cols[1].metric("Total number of suggestions", exp.aggregated_suggestions)

    total_sims = len(exp.per_sim_is_success)
    avg_hits = exp.aggregated_hits / total_sims if total_sims > 0 else 0
    cols[2].metric("Avg. hits per campaign", f"{avg_hits:.2f}")

    is_discrete = summary["is_discrete"]
    metric_name = "Accuracy" if is_discrete else "MAE"

    hits_per_sim = [sum(s) for s in exp.per_sim_n_hits]
    best_idx = np.argmax(hits_per_sim)
    worst_idx = np.argmin(hits_per_sim)

    # Success needs a hit threshold to be measured against
    if threshold is not None:
        success_count = sum(1 for s in exp.per_sim_is_success if s)
        success_rate = success_count / total_sims if total_sims > 0 else 0
        cols[3].metric("Success rate", f"{success_rate:.1%}")

        convergence_iterations, _, _ = al_plots._compute_convergence_stats(
            per_sim_n_hits=exp.per_sim_n_hits,
            n_hits_threshold=threshold,
        )
        perf_summary_chart = al_plots.build_performance_summary_chart(
            success_count=success_count,
            total_runs=total_sims,
            convergence_iterations=convergence_iterations,
            max_iterations=max((len(m) for m in exp.per_sim_metrics_total), default=0),
        )
        st.altair_chart(perf_summary_chart.interactive(), width="stretch")

    st.subheader("Target discovery progress")
    cum_success_chart = al_plots.build_mean_cumulative_successes_chart(
        per_sim_n_hits=exp.per_sim_n_hits,
        per_sim_is_success=exp.per_sim_is_success,
        target_threshold=threshold,
    )
    st.altair_chart(cum_success_chart.interactive(), width="stretch")

    st.subheader("Model performance evolution")
    metric_evo_chart = al_plots.build_mean_metric_evolution_chart(
        per_sim_metrics_total=exp.per_sim_metrics_total,
        best_run_metrics=exp.per_sim_metrics_total[best_idx],
        worst_run_metrics=exp.per_sim_metrics_total[worst_idx],
        metric_name=metric_name,
    )
    st.altair_chart(metric_evo_chart.interactive(), width="stretch")

    # Added: chart with suggestions metric overlay if available
    if exp.per_sim_metrics_suggestions:
        st.subheader("Model performance evolution (Suggestions) — mean per simulation")
        # Build a custom overlay chart?
        # Actually al_plots.build_mean_metric_evolution_chart doesn't seem to support suggestions
        # But we can at least show the mean suggestions metric
        suggestions_chart = al_plots.build_mean_metric_evolution_chart(
            per_sim_metrics_total=exp.per_sim_metrics_suggestions,
            best_run_metrics=exp.per_sim_metrics_suggestions[best_idx],
            worst_run_metrics=exp.per_sim_metrics_suggestions[worst_idx],
            metric_name=f"{metric_name} (Suggestions)",
        )
        st.altair_chart(suggestions_chart.interactive(), width="stretch")

    unique_hits, _ = _extract_unique_hits([exp])
    dataset_sequences = _load_pool(exp.dataset_id.name)
    projection_result = _load_projection(
        dataset_id=exp.dataset_id, embedder_name=exp.embedder
    )
    if projection_result is None:
        st.warning("Projection not available for this experiment.")
        return
    else:
        chart_projection = BiocentralChart.projection_result(
            projection_result=projection_result,
            dataset=dataset_sequences,
            highlight_ids=unique_hits,
            highlight_name="Unique Hits",
        )
        st.altair_chart(chart_projection.chart.interactive(), width="stretch")


def _render_single_simulation(
    exp: DashboardExperimentData, sim_data, summary: dict
) -> None:
    stop_reasons = sim_data.stop_reasons

    cols_row1 = st.columns(2)
    cols_row1[0].metric("Seed", sim_data.seed)
    if sim_data.n_hits_threshold is not None:
        cols_row1[1].metric("Success", sim_data.is_success)
    st.metric("Primary Stop Reason", stop_reasons[0])

    metric_name = "Accuracy" if summary["is_discrete"] else "MAE"

    n_iteration_hits = list(map(len, sim_data.iteration_hits))

    chart_metric = al_plots.build_metric_evolution_chart(
        iteration_metrics_total=sim_data.iteration_metrics_total,
        iteration_metrics_suggestions=sim_data.iteration_metrics_suggestions,
        n_iteration_hits=n_iteration_hits,
        metric_name=metric_name,
    )

    chart_failures = al_plots.build_consecutive_failures_chart(
        iteration_consecutive_failures=sim_data.iteration_consecutive_failures,
        stop_reasons=stop_reasons,
    )

    chart_successes = al_plots.build_n_hits_chart(n_iteration_hits=n_iteration_hits)

    col_left, col_right = st.columns(2)
    with col_left:
        st.altair_chart(chart_metric.interactive(), width="stretch")
        st.altair_chart(chart_failures.interactive(), width="stretch")
    with col_right:
        st.altair_chart(chart_successes.interactive(), width="stretch")

    # Projection Chart
    projection_result = _load_projection(
        dataset_id=exp.dataset_id, embedder_name=exp.embedder
    )
    if projection_result is None:
        st.warning("Projection not available for this experiment.")
        return
    else:
        # Projection Chart Navigation
        dataset_sequences = _load_pool(exp.dataset_id.name)

        # Create options: Full dataset + each iteration
        projection_options = ["Full Dataset"] + [
            f"Iteration {i + 1}" for i in range(len(sim_data.iteration_hits))
        ]

        # Initialize session state for projection selection if not exists
        if "projection_view_idx" not in st.session_state:
            st.session_state.projection_view_idx = 0

        iteration_projections = []
        all_iteration_hits = set()
        for iteration_hits in sim_data.iteration_hits:
            for hit in iteration_hits:
                all_iteration_hits.add(hit)
            chart_projection = BiocentralChart.projection_result(
                projection_result=projection_result,
                dataset=dataset_sequences,
                highlight_ids=set(iteration_hits),
                highlight_name="Iteration Hits",
            )
            iteration_projections.append(chart_projection)

        chart_projection_full = BiocentralChart.projection_result(
            projection_result=projection_result,
            dataset=dataset_sequences,
            highlight_ids=set(all_iteration_hits),
            highlight_name="All Hits",
        )
        projections = [chart_projection_full] + iteration_projections

        # Navigation controls
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button(
                "← Previous", disabled=st.session_state.projection_view_idx == 0
            ):
                st.session_state.projection_view_idx -= 1
                st.rerun()

        with col2:
            project_view_idx = st.session_state.projection_view_idx
            chart_title = projection_options[project_view_idx]
            st.markdown(f"**{chart_title}**")

        with col3:
            if st.button(
                "Next →",
                disabled=st.session_state.projection_view_idx
                == len(projection_options) - 1,
            ):
                st.session_state.projection_view_idx += 1
                st.rerun()

        project_view_idx = st.session_state.projection_view_idx
        chart_projection = projections[project_view_idx]

        st.altair_chart(chart_projection.chart.interactive(), width="stretch")

    with st.expander("Raw stats"):
        st.markdown(f"- **Stop reasons:** {stop_reasons}")
        st.markdown(f"- **Total iterations:** {sim_data.iteration_results_count}")
        st.markdown(
            f"- **iteration_metrics_total:** `{sim_data.iteration_metrics_total}`"
        )
        st.markdown(
            f"- **iteration_metrics_suggestions:** `{sim_data.iteration_metrics_suggestions}`"
        )
        st.markdown(
            f"- **iteration_consecutive_failures:** `{sim_data.iteration_consecutive_failures}`"
        )
        st.markdown(f"- **n_iteration_hits:** `{n_iteration_hits}`")
        st.markdown(f"- **iteration_hits:** `{sim_data.iteration_hits}`")


def _extract_consecutive_failures_threshold(stop_reasons: list[str]) -> int | None:
    for reason in stop_reasons:
        if "consecutive failures" in reason.lower():
            match = re.search(r"\((\d+)\)", reason)
            if match:
                return int(match.group(1))
    return None


def _extract_unique_hits(exps: list[DashboardExperimentData]) -> tuple[set[str], float]:
    """Get all unique hits and the std. dev. of all iteration hits found via bootstrapping."""
    all_hits = sum(e.aggregated_hits for e in exps)
    all_iteration_hits = []
    for e in exps:
        single_sim_res = e.single_sims
        for single_sim in single_sim_res:
            all_iteration_hits.extend(single_sim.iteration_hits)

    all_hits_flattened = [hit for sublist in all_iteration_hits for hit in sublist]
    all_unique_hits = set(all_hits_flattened)
    assert len(all_hits_flattened) == all_hits, (
        f"Number of iteration hits ({len(all_iteration_hits)}) "
        f"does not match total hits ({all_hits})"
    )  # TODO: Optimize compression

    # Bootstrap resampling over unique hits found to calculate std dev
    n_bootstrap = 30
    bootstrapped_n_unique_hits = []
    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = rng.choice(
            len(all_iteration_hits), size=len(all_iteration_hits), replace=True
        )
        bootstrap_iteration_hits = [all_iteration_hits[i] for i in bootstrap_sample]
        bootstrap_iteration_hits_flattened = [
            hit for sublist in bootstrap_iteration_hits for hit in sublist
        ]
        bootstrap_unique_hits = set(bootstrap_iteration_hits_flattened)
        bootstrap_n_unique_hits = len(bootstrap_unique_hits)
        bootstrapped_n_unique_hits.append(bootstrap_n_unique_hits)

    bootstrap_std = float(np.std(bootstrapped_n_unique_hits))

    return all_unique_hits, bootstrap_std


def _get_round_errors(
    exp: DashboardExperimentData, subset: str, error_metric: str
) -> list[list[float]]:
    """Per-round error of each replicate on its suggestions (`subset="suggestions"`) or every scored candidate (`subset="scored"`).
    Accuracy for discrete targets.
    """
    if exp.per_sim_round_errors is None:
        if subset == "suggestions":
            return exp.per_sim_metrics_suggestions
        return exp.per_sim_metrics_total
    return exp.per_sim_round_errors[f"{error_metric.lower()}_{subset}"]


def _collect_replicate_metrics(
    exp: DashboardExperimentData, error_metric: str
) -> dict[str, Sequence[float | None]]:
    """Each replicate's total hits, hit rate, and best label (unless target is discrete),
    and the final round's prediction metrics, with `error_metric` as error. A ranking
    metric is None for a replicate whose predictions cannot rank."""
    hits = [sum(n_hits) for n_hits in exp.per_sim_n_hits]
    metrics: dict[str, Sequence[float | None]] = {
        "total_hits": hits,
        "hit_rate": [
            n_hits / n_suggestions if n_suggestions else 0.0
            for n_hits, n_suggestions in zip(hits, exp.per_sim_n_suggestions)
        ],
    }
    if exp.per_sim_best_label is not None:
        metrics["best_label"] = exp.per_sim_best_label
    final_round_metrics = exp.per_sim_final_round_metrics or []
    for name in (final_round_metrics or [{}])[0]:
        if name in ("mae", "rmse") and name != error_metric.lower():
            continue
        key = "final_error" if name in ("mae", "rmse") else f"final_{name}"
        metrics[key] = [replicate[name] for replicate in final_round_metrics]
    return metrics


def _build_replicate_frame(
    exps: list[DashboardExperimentData], error_metric: str
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "encoder": _get_encoder_label(exp.embedder),
                "model": exp.model,
                "value": v,
            }
            for exp in exps
            for metric, values in _collect_replicate_metrics(exp, error_metric).items()
            for v in values
            if v is not None
        ],
        columns=["metric", "encoder", "model", "value"],
    )


def _average_per_round(
    per_sim: Sequence[Sequence[float]], cumulative: bool
) -> list[float]:
    """Per-round mean over replicates. Stopped replicates' totals are kept in cumulative series."""
    n_rounds = max(map(len, per_sim), default=0)
    padded = []
    for values in per_sim:
        values = np.cumsum(values).tolist() if cumulative else list(values)
        fill = values[-1] if cumulative and values else np.nan
        padded.append(values + [fill] * (n_rounds - len(values)))
    return np.nanmean(np.array(padded, dtype=float), axis=0).tolist()


def _build_rounds_frame(
    subset: list[DashboardExperimentData],
    per_sim: Callable[[DashboardExperimentData], Sequence[Sequence[float]] | None],
    cumulative: bool,
) -> pd.DataFrame:
    """Each campaign's mean per round, leaving out campaigns without the series."""
    return pd.DataFrame(
        [
            {
                "model": exp.model,
                "encoder": _get_encoder_label(exp.embedder),
                "round": i + 1,
                "value": v,
            }
            for exp in subset
            if (runs := per_sim(exp)) is not None
            for i, v in enumerate(_average_per_round(runs, cumulative))
        ],
        columns=["model", "encoder", "round", "value"],
    )


def _build_calibration_frame(exps: list[DashboardExperimentData]) -> pd.DataFrame:
    """Observed coverage per calibration level, averaged over replicates of each campaign."""
    return pd.DataFrame(
        [
            {
                "model": exp.model,
                "encoder": _get_encoder_label(exp.embedder),
                "expected": level,
                "observed": observed,
            }
            for exp in exps
            if exp.per_sim_calibration is not None
            for level, observed in zip(
                UNCERTAINTY_CALIBRATION_LEVELS, np.mean(exp.per_sim_calibration, axis=0)
            )
        ],
        columns=["model", "encoder", "expected", "observed"],
    )


def _describe_ndcg_cutoffs(exps: list[DashboardExperimentData], percentage: int) -> str:
    """The n of NDCG@percentage% over the experiments' replicates, as a value or a range."""
    ns = sorted(
        {
            percentage_cutoff(n_scored, percentage)
            for exp in exps
            for n_scored in exp.per_sim_final_round_scored
        }
    )
    return f"n = {ns[0]}" if ns[0] == ns[-1] else f"n = {ns[0]}–{ns[-1]}"


def _render_campaigns_tab(
    subset: list[DashboardExperimentData], error_metric: str
) -> None:
    random = [exp for exp in subset if exp.model == "RANDOM"]
    surrogates = [exp for exp in subset if exp.model != "RANDOM"]
    maximize = subset[0].summary["optimization_mode"] == "MAXIMIZE"
    metric_name = "Accuracy" if subset[0].summary["is_discrete"] else error_metric
    hit_titles = {
        "total_hits": "Total Hits",
        "hit_rate": "Hit Rate",
        "best_label": "Maximum Fitness Achieved"
        if maximize
        else "Minimum Label Achieved",
    }
    prediction_titles = {
        "final_error": metric_name,
        "final_spearman": "Spearman Correlation",
        "final_ndcg@10": "NDCG@10",
        "final_ndcg@96": "NDCG@96",
        "final_ndcg@1%": f"NDCG@1% ({_describe_ndcg_cutoffs(subset, 1)})",
        "final_ndcg@10%": f"NDCG@10% ({_describe_ndcg_cutoffs(subset, 10)})",
    }
    replicates = _build_replicate_frame(surrogates, error_metric)
    baselines = _build_replicate_frame(random, error_metric)
    present = set(replicates["metric"])
    n_replicates = sorted({len(exp.per_sim_n_hits) for exp in subset})
    random_note = (
        " Dashed line: the random model, which runs with one-hot encoding only."
        if random
        else ""
    )

    st.subheader("Hits")
    st.altair_chart(
        al_plots.build_grouped_metric_bars(
            replicates,
            baselines,
            {
                metric: title
                for metric, title in hit_titles.items()
                if metric in present
            },
            ENCODER_ORDER,
            columns=3,
        )
    )
    st.caption(
        f"Bars: mean over each campaign's {'/'.join(map(str, n_replicates))} replicates, "
        "dots: the replicates, whiskers: standard error." + random_note
    )

    prediction_titles = {
        metric: title
        for metric, title in prediction_titles.items()
        if metric in present
    }
    if prediction_titles:
        st.subheader("Predictions on the Whole Landscape\\*")
        st.altair_chart(
            al_plots.build_grouped_metric_bars(
                replicates, baselines, prediction_titles, ENCODER_ORDER, columns=2
            )
        )
        st.caption(
            "\\* For now on the variants the final round's model scored rather than the "
            "whole landscape: with pool candidates a random sample of up to 1,000 unseen "
            "variants, with generated candidates the generated mutants. NDCG ranks them by "
            "prediction, with the labels min-max scaled to [0, 1] as gains; NDCG@k% looks at "
            "the top k% of them. Spearman and NDCG leave out replicates whose predictions take "
            "at most two values (the GP with the length embedder). Bars, dots and whiskers as "
            "above." + random_note
        )

    st.subheader("Cumulative Hits")
    st.altair_chart(
        al_plots.build_rounds_by_model_chart(
            _build_rounds_frame(
                surrogates, lambda exp: exp.per_sim_n_hits, cumulative=True
            ),
            _build_rounds_frame(
                random, lambda exp: exp.per_sim_n_hits, cumulative=True
            ),
            "Cumulative hits (mean)",
            ENCODER_ORDER,
        )
    )

    if subset[0].per_sim_suggested_label_means is not None:
        st.subheader("Mean Label of Chosen Sequences")
        st.altair_chart(
            al_plots.build_rounds_by_model_chart(
                _build_rounds_frame(
                    surrogates,
                    lambda exp: exp.per_sim_suggested_label_means,
                    cumulative=False,
                ),
                _build_rounds_frame(
                    random,
                    lambda exp: exp.per_sim_suggested_label_means,
                    cumulative=False,
                ),
                "Mean label (mean)",
                ENCODER_ORDER,
            )
        )

    panels = {
        "Chosen sequences": lambda exp: _get_round_errors(
            exp, "suggestions", error_metric
        ),
        "Unseen variants": lambda exp: _get_round_errors(exp, "scored", error_metric),
    }

    def build_panel_frame(exps: list[DashboardExperimentData]) -> pd.DataFrame:
        return pd.concat(
            [
                _build_rounds_frame(exps, per_sim, cumulative=False).assign(panel=panel)
                for panel, per_sim in panels.items()
            ],
            ignore_index=True,
        )

    st.subheader(f"{metric_name} Evolution")
    st.altair_chart(
        al_plots.build_rounds_by_model_chart(
            build_panel_frame(surrogates),
            build_panel_frame(random),
            f"{metric_name} (mean)",
            ENCODER_ORDER,
            log_y=metric_name != "Accuracy",
        )
    )
    st.caption(
        ("y-axis logarithmic. " if metric_name != "Accuracy" else "")
        + "Unseen variants: the candidates each round's model scored before choosing the "
        "sequences to measure. With pool candidates, a random sample of up to 1,000 variants the model "
        "has not seen; with generated candidates, mutants of the variants it has measured. "
        "The server does not record the error on the variants the model has learned."
    )

    calibration = _build_calibration_frame(surrogates)
    if not calibration.empty:
        st.subheader("Uncertainty Calibration")
        st.altair_chart(al_plots.build_calibration_chart(calibration, ENCODER_ORDER))
        st.caption(
            "Share of the candidates each round's model scored whose label lies within the "
            "central interval of the predicted normal distribution, pooled over rounds and "
            "averaged over replicates. Below the diagonal, the uncertainties are too narrow. "
            "The random model's uncertainty is a constant, so it is left out."
        )


def _render_header(exp: DashboardExperimentData) -> None:
    summary = exp.summary
    st.markdown(f"### `{exp.name}`")
    cols_row1 = st.columns(3)
    cols_row1[0].metric("Embedder", _get_encoder_label(exp.embedder))
    cols_row1[1].metric("Model", exp.model)
    cols_row1[2].metric("Mode", str(summary["optimization_mode"]))

    cols_row2 = st.columns(3)
    cols_row2[0].metric("Sims", summary["n_simulations"])
    if summary["n_hits_threshold"] is not None:
        cols_row2[1].metric(
            "Successful", f"{summary['n_successful']}/{summary['n_simulations']}"
        )
        cols_row2[2].metric("Target threshold", summary["n_hits_threshold"])

    setup = exp.setup
    cols_setup = st.columns(4)
    cols_setup[0].metric("Starting set", setup.n_start)
    cols_setup[1].metric("Suggestions per round", setup.n_suggestions_per_iteration)
    cols_setup[2].metric("Candidates", _describe_candidate_source(setup))
    cols_setup[3].metric(
        "Delta embeddings", "On" if setup.use_delta_embeddings else "Off"
    )
    st.caption(_describe_stopping_criteria(setup))
    if summary.get("discrete_targets"):
        st.caption(f"Discrete targets: {', '.join(summary['discrete_targets'])}")


def _render_dataset_tab(selected_dataset: str, exps):
    pool = _load_pool(selected_dataset)
    definition = ALSimulatorDataset[selected_dataset].definition()
    exp = exps[0]
    st.metric(
        "Number of hits",
        value=_summarize_potential_hits(exp, len(pool)),
        help=_describe_hits(exp),
    )

    labels = [data_point.label for data_point in pool if data_point.label is not None]
    maximize = exp.summary["optimization_mode"] == "MAXIMIZE"
    try:
        values = [float(label) for label in labels]
    except ValueError:
        values = None
    threshold = None if values is None else _compute_hit_threshold(exp, values)
    # Labels that are not numbers are categories named after themselves
    label_names = definition.label_names or (
        None if values is not None else {label: label for label in set(labels)}
    )

    st.markdown("**Label Distribution**")
    st.altair_chart(
        al_plots.build_label_distribution_chart(
            labels, threshold, maximize, label_names
        ),
        width="stretch",
    )
    notes = []
    if values is not None and label_names is None:
        notes.append(
            f"Mean {np.mean(values):.2f} · median {np.median(values):.2f} · "
            f"SD {np.std(values, ddof=1):.2f}"
        )
    if threshold is not None:
        value = (
            ""
            if label_names
            else f" at {np.format_float_positional(threshold, precision=4, fractional=False, trim='-')}"
        )
        notes.append(f"Dashed line: hit threshold{value}. {_describe_hits(exp)}")
    if notes:
        st.caption("  \n".join(notes))

    depths = _load_depths(selected_dataset)
    if depths is not None:
        st.markdown(
            "**Mutational Depth**",
            help="Substitutions of each variant from the dataset's parent sequence.",
        )
        st.altair_chart(
            al_plots.build_mutation_depth_chart(
                {
                    "Full landscape": list(depths.values()),
                    "Hits": [depths[hit] for hit in exp.potential_hits],
                }
            ),
            width="stretch",
        )


def main():
    # ---------------------------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------------------------
    st.sidebar.header("Simulation Results")

    # ---------------------------------------------------------------------------
    # Main area
    # ---------------------------------------------------------------------------
    st.title("Active Learning Simulation Dashboard")

    _available_runs = _get_run_files()
    all_compressed_data = _load_compressed(_available_runs)
    if len(all_compressed_data) > 0:
        st.sidebar.success("Loaded all results.")
    else:
        st.error(
            f"Compressed data files not found at `{RESULTS_DIR}`. "
            "Run `uv run python -m biocentral_al_sim.compress_reports` first."
        )
        st.stop()

    st.sidebar.subheader("Select Run")
    selected_run = st.sidebar.selectbox(
        "Select Run", sorted(all_compressed_data.keys()), label_visibility="collapsed"
    )

    compressed_data = all_compressed_data[selected_run]
    by_dataset: dict[str, list[DashboardExperimentData]] = {}
    for exp in compressed_data.experiments:
        by_dataset.setdefault(exp.dataset_id.name, []).append(exp)

    st.sidebar.subheader("Select Dataset")
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset", sorted(by_dataset), label_visibility="collapsed"
    )
    exps = by_dataset[selected_dataset]

    subset = _filter_experiments(exps)
    st.sidebar.subheader("Display")
    error_metric = st.sidebar.radio("Error metric", ["RMSE", "MAE"], horizontal=True)

    main_tabs = st.tabs(["Dataset statistics", "Campaigns", "Single campaign"])

    with main_tabs[0]:
        _render_dataset_tab(selected_dataset, exps)

    with main_tabs[1]:
        _render_campaigns_tab(subset, error_metric)

    with main_tabs[2]:
        by_label = _index_by_label(subset)
        exp = by_label[st.selectbox("Select Experiment", list(by_label))]

        _render_header(exp)

        sub_tabs = st.tabs(["Aggregate", "Per-Simulation Drill-down"])
        with sub_tabs[0]:  # Simulations Aggregate
            _render_simulations_aggregate(exp)
        with sub_tabs[1]:  # Per-simulation drill-down
            sim_labels = {s.label: s for s in exp.single_sims}
            selected_label = st.selectbox("Simulation run", sorted(sim_labels.keys()))
            sim_data = sim_labels[selected_label]
            _render_single_simulation(exp, sim_data, exp.summary)


if __name__ == "__main__":
    main()
