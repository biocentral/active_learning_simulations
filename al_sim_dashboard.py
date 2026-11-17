"""Streamlit dashboard for active learning simulation results.

Compress the results, then run the dashboard:
    uv run python -m biocentral_al_sim.compress_reports
    uv run streamlit run al_sim_dashboard.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from biocentral_api import ProjectionResult
from biocentral_vis import BiocentralChart
from biotrainer_core.data_classes import SequenceData

from biocentral_al_sim import plots as al_plots
from biocentral_al_sim.dashboard_models import (
    DashboardCampaignSetup,
    DashboardCompressedData,
    DashboardExperimentData,
)
from biocentral_al_sim.datasets import ALSimulatorDataset
from biocentral_al_sim.paths import PROJECTIONS_DIR, RESULTS_DIR
from biocentral_al_sim.settings import THROUGHPUTS
from biocentral_al_sim.splits import substitutions

st.set_page_config(page_title="AL Simulation Dashboard", layout="wide")


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


def _hit_definition(exp: DashboardExperimentData) -> str:
    match exp.summary["optimization_mode"]:
        case "MAXIMIZE":
            return f"Pool members in the top {exp.setup.hit_percentile:g}% of the pool's labels."
        case "MINIMIZE":
            return f"Pool members in the bottom {exp.setup.hit_percentile:g}% of the pool's labels."
        case _:
            return "Pool members that meet the target."


def _hit_threshold(exp: DashboardExperimentData, labels: list[float]) -> float | None:
    """Compute hit threshold of annotations from servec's percentile rule."""
    match exp.summary["optimization_mode"]:
        case "MAXIMIZE":
            return float(np.percentile(labels, 100 - exp.setup.hit_percentile))
        case "MINIMIZE":
            return float(np.percentile(labels, exp.setup.hit_percentile))
        case _:
            return None


def _potential_hits_summary(exp: DashboardExperimentData, n_pool: int) -> str:
    n_potential_hits = len(exp.potential_hits)
    return f"{n_potential_hits} ({100 * n_potential_hits / n_pool:.2f} % of the pool)"


def _candidate_source(setup: DashboardCampaignSetup) -> str:
    if not setup.generate_candidates:
        return "Pool"
    plural = "" if setup.n_substitutions == 1 else "s"
    return f"Generated ({setup.n_substitutions} substitution{plural})"


def _stopping_criteria(setup: DashboardCampaignSetup) -> str:
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


def _throughput(setup: DashboardCampaignSetup) -> str:
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


def _arm(exp: DashboardExperimentData) -> tuple:
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
        list(dict.fromkeys(_throughput(e.setup) for e in by_size)),
        help="Starting set + rounds × sequences chosen per round",
    )
    experiments = [e for e in experiments if _throughput(e.setup) == throughput]

    by_source = sorted(experiments, key=lambda e: e.setup.n_substitutions or 0)
    source = st.sidebar.radio(
        "Candidate generation",
        list(dict.fromkeys(_candidate_source(e.setup) for e in by_source)),
    )
    experiments = [e for e in experiments if _candidate_source(e.setup) == source]

    if not any(e.setup.use_delta_embeddings for e in experiments):
        return experiments
    delta = st.sidebar.radio(
        "Delta embeddings",
        ["Off", "On"],
        horizontal=True,
    )
    if delta == "Off":
        return [e for e in experiments if not e.setup.use_delta_embeddings]
    with_delta = {_arm(e) for e in experiments if e.setup.use_delta_embeddings}
    return [
        e
        for e in experiments
        if e.setup.use_delta_embeddings or _arm(e) not in with_delta
    ]


def _by_label(
    exps: list[DashboardExperimentData],
) -> dict[str, DashboardExperimentData]:
    """Experiments by model and embedder. Ambiguous labels fall back to the file name."""
    exps = sorted(exps, key=lambda e: (e.model, e.embedder))
    labels = [
        f"{e.model} · {e.embedder}"
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


def _extract_average_hits_per_campaign(
    exps: list[DashboardExperimentData],
) -> tuple[float, float]:
    all_single_sims = []
    for e in exps:
        all_single_sims.extend(e.single_sims)
    n_all_hits_list = [
        sum(list(map(len, single_sim.iteration_hits))) for single_sim in all_single_sims
    ]
    total_hits_mean = round(float(np.mean(n_all_hits_list)), 3)
    total_hits_std = float(np.std(n_all_hits_list))
    return total_hits_mean, total_hits_std


def _render_inner_simulation_comparison(subset: list[DashboardExperimentData]) -> None:
    mode = st.radio(
        "Compare by",
        ["Surrogate Model", "Embedder"],
        key="InnerSimulationRadio",
        horizontal=True,
    )

    group_by = "model" if mode == "Surrogate Model" else "embedder"

    groups: dict[str, list[DashboardExperimentData]] = {}
    for exp in subset:
        if group_by == "embedder" and exp.model == "RANDOM":
            continue  # Exclude random surrogate models from embedder comparison
        key = getattr(exp, group_by)
        if key not in groups:
            groups[key] = []
        groups[key].append(exp)

    comparison_data = []
    cross_cum_inputs = {}
    cross_metric_inputs = {}
    cross_targets = {}
    metric_names = set()

    for key, exps in groups.items():
        all_hits = sum(e.aggregated_hits for e in exps)
        all_suggestions = sum(e.aggregated_suggestions for e in exps)
        all_unique_hits, unique_hits_std = _extract_unique_hits(exps)
        findable_hits = set(exps[0].potential_hits) - set.intersection(
            *(set(e.start_set_hits) for e in exps)
        )

        all_is_success = [s for e in exps for s in e.per_sim_is_success]
        total_campaigns = len(all_is_success)

        avg_hits, _ = _extract_average_hits_per_campaign(exps)
        row = {
            mode: key,
            "Total Hits": all_hits,
            "Unique Hits / Findable": f"{len(all_unique_hits)} / {len(findable_hits)}",
            "Unique Hits Std. Dev.": f"{unique_hits_std:.2f}",
            "Number of Campaigns": total_campaigns,
            "Total Suggestions": all_suggestions,
            "Hit Rate": f"{all_hits / all_suggestions:.2%}"
            if all_suggestions > 0
            else "0%",
            "Avg. Hits per Campaign": f"{avg_hits:.2f}",
        }

        # Success needs a hit threshold to be measured against
        if any(e.summary["n_hits_threshold"] is not None for e in exps):
            conv_speeds = []
            for e in exps:
                target_threshold = e.summary["n_hits_threshold"]
                for i, is_succ in enumerate(e.per_sim_is_success):
                    if is_succ:
                        cumsum = np.cumsum(e.per_sim_n_hits[i])
                        converged_at = np.where(cumsum >= target_threshold)[0]
                        if len(converged_at) > 0:
                            conv_speeds.append(int(converged_at[0]) + 1)

            avg_conv = np.mean(conv_speeds) if conv_speeds else None
            row["Percentage of successful campaigns"] = (
                f"{sum(all_is_success) / total_campaigns:.1%}"
                if total_campaigns > 0
                else "0.0%"
            )
            row["Avg. Convergence Iteration"] = (
                f"{avg_conv:.2f}" if avg_conv is not None else "N/A"
            )
        comparison_data.append(row)

    if group_by == "embedder":
        st.markdown("*Note: Random surrogate models are excluded from the comparison.*")
    st.table(pd.DataFrame(comparison_data))
    st.caption(
        "Findable hits leave out potential hits that every campaign of the group started "
        "with. Totals and unique hits grow with the number of campaigns, so compare them "
        "only between groups of equal size."
    )

    st.subheader("Visual Comparison")
    for exp in subset:
        cross_cum_inputs[exp.name] = exp.per_sim_n_hits
        cross_metric_inputs[exp.name] = exp.per_sim_metrics_total
        if exp.summary["n_hits_threshold"] is not None:
            cross_targets[exp.name] = exp.summary["n_hits_threshold"]
        metric_names.add("Accuracy" if exp.summary["is_discrete"] else "MAE")

    metric_name = ", ".join(sorted(metric_names)) if metric_names else "Metric"

    st.subheader("Cumulative target successes — mean per experiment")
    st.altair_chart(
        al_plots.build_cross_experiment_cumulative_chart(
            cross_cum_inputs, cross_targets
        ).interactive(),
        width="stretch",
    )
    st.subheader(f"Model performance ({metric_name}) — mean per experiment")
    st.altair_chart(
        al_plots.build_cross_experiment_metric_chart(
            cross_metric_inputs, metric_name
        ).interactive(),
        width="stretch",
    )


def _render_header(exp: DashboardExperimentData) -> None:
    summary = exp.summary
    st.markdown(f"### `{exp.name}`")
    cols_row1 = st.columns(3)
    cols_row1[0].metric("Embedder", exp.embedder)
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
    cols_setup[2].metric("Candidates", _candidate_source(setup))
    cols_setup[3].metric(
        "Delta embeddings", "On" if setup.use_delta_embeddings else "Off"
    )
    st.caption(_stopping_criteria(setup))
    if summary.get("discrete_targets"):
        st.caption(f"Discrete targets: {', '.join(summary['discrete_targets'])}")


def _render_dataset_tab(selected_dataset: str, exps):
    pool = _load_pool(selected_dataset)
    definition = ALSimulatorDataset[selected_dataset].definition()
    exp = exps[0]
    st.metric(
        "Number of hits",
        value=_potential_hits_summary(exp, len(pool)),
        help=_hit_definition(exp),
    )

    labels = [data_point.label for data_point in pool]
    maximize = exp.summary["optimization_mode"] == "MAXIMIZE"
    try:
        values = [float(label) for label in labels]
    except (TypeError, ValueError):
        values = None
    threshold = None if values is None else _hit_threshold(exp, values)
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
        notes.append(f"Dashed line: hit threshold{value}. {_hit_definition(exp)}")
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

    main_tabs = st.tabs(["Dataset statistics", "Campaigns", "Single campaign"])

    with main_tabs[0]:
        _render_dataset_tab(selected_dataset, exps)

    with main_tabs[1]:
        _render_inner_simulation_comparison(subset)

    with main_tabs[2]:
        by_label = _by_label(subset)
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
