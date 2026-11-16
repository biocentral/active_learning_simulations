"""Streamlit dashboard for active learning simulation results.

Compress the results, then run the dashboard:
    uv run python -m biocentral_al_sim.compress_reports
    uv run streamlit run al_sim_dashboard.py
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
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

st.set_page_config(page_title="AL Simulation Dashboard", layout="wide")


@st.cache_data(show_spinner=True)
def _get_run_files():
    return list(RESULTS_DIR.glob("compressed_dashboard_data_*.json"))


@st.cache_resource(show_spinner=True)
def _load_pool(selected_dataset: str) -> list[SequenceData]:
    """Sequences the dataset's campaigns drew from."""
    return ALSimulatorDataset[selected_dataset].pool()


@st.cache_data(show_spinner=True)
def _load_pool_table(selected_dataset: str) -> pd.DataFrame:
    df = pd.DataFrame([seq.model_dump() for seq in _load_pool(selected_dataset)])

    # Convert label to float if possible
    if "label" in df.columns:

        def try_float_conversion(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return val

        df["label"] = df["label"].apply(try_float_conversion)

    return df.drop(columns=["attributes", "embedding", "mask", "set"], errors="ignore")


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
    """Narrow the run to one throughput, candidate source and embedding variant."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Campaign setup")

    by_size = sorted(
        experiments,
        key=lambda e: (e.setup.n_start, e.setup.n_suggestions_per_iteration),
    )
    throughput = st.sidebar.radio(
        "Throughput",
        list(dict.fromkeys(_throughput(e.setup) for e in by_size)),
        help="Starting set + rounds × suggestions per round",
    )
    experiments = [e for e in experiments if _throughput(e.setup) == throughput]

    by_source = sorted(experiments, key=lambda e: e.setup.n_substitutions or 0)
    source = st.sidebar.radio(
        "Candidates",
        list(dict.fromkeys(_candidate_source(e.setup) for e in by_source)),
        help="Drawn from the pool, or generated by mutating measured variants",
    )
    experiments = [e for e in experiments if _candidate_source(e.setup) == source]

    if not any(e.setup.use_delta_embeddings for e in experiments):
        return experiments
    delta = st.sidebar.radio(
        "Delta embeddings",
        ["Off", "On"],
        horizontal=True,
        help="On shows the delta run of every embedder and model that has one, and the "
        "plain run of the rest (baseline embedders, the random model).",
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

    cols = st.columns(4)
    cols[0].metric("Total number of hits", exp.aggregated_hits)
    cols[1].metric("Total number of suggestions", exp.aggregated_suggestions)

    total_sims = len(exp.per_sim_is_success)
    avg_hits = exp.aggregated_hits / total_sims if total_sims > 0 else 0
    success_rate = (
        sum(1 for s in exp.per_sim_is_success if s) / total_sims
        if total_sims > 0
        else 0
    )

    cols[2].metric("Avg. hits per campaign", f"{avg_hits:.2f}")
    cols[3].metric("Success rate", f"{success_rate:.1%}")

    summary = exp.summary
    is_discrete = summary["is_discrete"]
    metric_name = "Accuracy" if is_discrete else "MAE"

    # We need to compute stats for the charts
    max_iterations = max((len(m) for m in exp.per_sim_metrics_total), default=0)
    success_count = sum(1 for s in exp.per_sim_is_success if s)

    convergence_iterations, _, _ = al_plots._compute_convergence_stats(
        per_sim_n_hits=exp.per_sim_n_hits,
        n_hits_threshold=summary["n_hits_threshold"],
    )

    hits_per_sim = [sum(s) for s in exp.per_sim_n_hits]
    best_idx = np.argmax(hits_per_sim)
    worst_idx = np.argmin(hits_per_sim)

    perf_summary_chart = al_plots.build_performance_summary_chart(
        success_count=success_count,
        total_runs=len(exp.per_sim_is_success),
        convergence_iterations=convergence_iterations,
        max_iterations=max_iterations,
    )
    st.altair_chart(perf_summary_chart.interactive(), width="stretch")

    st.subheader("Target discovery progress")
    cum_success_chart = al_plots.build_mean_cumulative_successes_chart(
        per_sim_n_hits=exp.per_sim_n_hits,
        per_sim_is_success=exp.per_sim_is_success,
        target_threshold=summary["n_hits_threshold"],
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
    st.header("Comparison")
    mode = st.radio(
        "Compare by",
        ["Surrogate Model", "Embedder"],
        key="InnerSimulationRadio",
        horizontal=True,
    )

    group_by = "model" if mode == "Surrogate Model" else "embedder"

    groups: dict[str, list[DashboardExperimentData]] = {}
    for exp in subset:
        key = getattr(exp, group_by)
        if key not in groups:
            groups[key] = []
        if group_by == "embedder" and exp.model == "RANDOM":
            continue  # Exclude random surrogate models from embedder comparison
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

        all_is_success = [s for e in exps for s in e.per_sim_is_success]
        total_campaigns = len(all_is_success)

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
        avg_hits, _ = _extract_average_hits_per_campaign(exps)
        comparison_data.append(
            {
                mode: key,
                "Total Hits": all_hits,
                "Unique Hits": len(all_unique_hits),
                "Unique Hits Std. Dev.": f"{unique_hits_std:.2f}",
                "Number of Campaigns": total_campaigns,
                "Total Suggestions": all_suggestions,
                "Hit Rate": f"{all_hits / all_suggestions:.2%}"
                if all_suggestions > 0
                else "0%",
                "Avg. Hits per Campaign": f"{avg_hits:.2f}",
                "Percentage of successful campaigns": f"{sum(all_is_success) / total_campaigns:.1%}"
                if total_campaigns > 0
                else "0.0%",
                "Avg. Convergence Iteration": f"{avg_conv:.2f}"
                if avg_conv is not None
                else "N/A",
            }
        )

    if group_by == "embedder":
        st.markdown("*Note: Random surrogate models are excluded from the comparison.*")
    st.table(pd.DataFrame(comparison_data))

    st.subheader("Visual Comparison")
    for exp in subset:
        cross_cum_inputs[exp.name] = exp.per_sim_n_hits
        cross_metric_inputs[exp.name] = exp.per_sim_metrics_total
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
    st.metric(
        "**Number of potential hits:**",
        value=_potential_hits_summary(exps[0], len(pool)),
        help=_hit_definition(exps[0]),
    )
    df = _load_pool_table(selected_dataset)
    df.insert(1, "potential_hit", df["seq_id"].isin(set(exps[0].potential_hits)))

    st.markdown(
        "**Pool**",
        help="Sequences the campaigns drew from, potential hits first.",
    )
    st.dataframe(df.sort_values("potential_hit", ascending=False))

    # Label distribution Chart
    biotrainer_chart = BiocentralChart.label_distribution(pool)
    st.markdown(
        "**Pool Label Distribution**",
        help="Distribution of all labels in the pool.",
    )
    chart = biotrainer_chart.chart

    chart = (
        chart.properties(width=1200, height=800)
        .configure_axis(labelFontSize=20, titleFontSize=28)
        .configure_axisX(labelAngle=-45, labelLimit=0, labelOverlap=False)
        .configure_legend(labelFontSize=24, titleFontSize=24)
        .configure_title(fontSize=24)
    )
    st.altair_chart(chart, width="stretch")


def _extract_cross_dataset_data(
    group_by: str,
    ds_groups: dict[str, list[DashboardExperimentData]],
    dict_constructor: Callable,
):
    cross_dataset_data = {}  # ds_name -> n unique hits
    all_groupable = {}  # Model Name / Embedder Name -> Experiments

    if group_by == "dataset":
        for ds_name, ds_exps in ds_groups.items():
            cross_dataset_data[ds_name] = dict_constructor(ds_exps)
    else:
        for ds_name, ds_exps in ds_groups.items():
            for exp in ds_exps:
                key = getattr(exp, group_by)
                if group_by == "embedder" and exp.model == "RANDOM":
                    continue  # Exclude random surrogate models from embedder comparison
                if key not in all_groupable:
                    all_groupable[key] = []
                all_groupable[key].append(exp)

        for key, exps in all_groupable.items():
            cross_dataset_data[key] = dict_constructor(exps)

    return cross_dataset_data


def _plot_cross_dataset_data(cross_dataset_data: dict, value_name: str):
    import altair as alt
    import pandas as pd

    # Prepare data for plotting with error bars
    plot_data = []
    for key, values in cross_dataset_data.items():
        if isinstance(values, dict) and "std" in values:
            # Extract the main metric (first key that's not 'std')
            metric_key = next(k for k in values if k != "std")
            mean_value = values[metric_key]
            std_value = values["std"]
            plot_data.append(
                {
                    "category": key,
                    "value": mean_value,
                    "lower": max(0, mean_value - std_value),
                    "upper": mean_value + std_value,
                    "std_value": std_value,
                }
            )
        else:
            plot_data.append(
                {"category": key, "value": values, "lower": values, "upper": values}
            )

    # Create DataFrame for the chart
    df = pd.DataFrame(plot_data)
    df = df.sort_values("value", ascending=False).reset_index(drop=True)
    category_sort = df["category"].tolist()

    # Create bar chart with error bars
    bars = (
        alt.Chart(df)
        .mark_bar(
            size=100,
            filled=True,
        )
        .encode(
            x=alt.X(
                "category:N",
                title=None,
                axis=alt.Axis(labelAngle=-45),
                sort=category_sort,
            ),
            y=alt.Y("value:Q", title=value_name),
            color=alt.Color(
                "category:N", scale=alt.Scale(scheme="tableau10"), legend=None
            ),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("value:Q", title=value_name, format=".2f"),
                alt.Tooltip("std_value:Q", title="Std Dev", format=".2f"),
            ],
        )
    )

    # Create error bars
    error_bars = (
        alt.Chart(df)
        .mark_errorbar(color="black")
        .encode(
            y=alt.Y("lower:Q", title=value_name),
            y2=alt.Y2("upper:Q", title=None),
            x=alt.X("category:N", sort=category_sort),
        )
    )

    chart = bars + error_bars

    chart = (
        chart.properties(width=1200, height=800)
        .configure_axis(labelFontSize=20, titleFontSize=28)
        .configure_axisX(labelAngle=-45, labelLimit=0, labelOverlap=False)
        .configure_legend(labelFontSize=24, titleFontSize=24)
        .configure_title(fontSize=24)
    )

    st.altair_chart(chart, width="stretch")


def _render_cross_dataset_comparison(
    ds_groups: dict[str, list[DashboardExperimentData]],
):
    st.markdown("## Cross-dataset comparison")
    mode = st.radio(
        "Compare by",
        ["Surrogate Model", "Embedder", "Dataset"],
        key="CrossDatasetRadio",
        horizontal=True,
    )

    group_by = "model" if mode == "Surrogate Model" else mode.lower()

    ### AVERAGE
    def average_dict_constructor(exps):
        average_hits, average_hits_std = _extract_average_hits_per_campaign(exps)
        return {"average_hits_per_campaign": average_hits, "std": average_hits_std}

    cross_dataset_data_average = _extract_cross_dataset_data(
        group_by, ds_groups, average_dict_constructor
    )

    match group_by:
        case "model":
            st.markdown(
                "**Average hits found by each surrogate model across all datasets:**",
                help="Measure of power of different surrogate models.",
            )
            st.markdown(
                "*Note: Random surrogate models are excluded from the comparison.*"
            )
        case "embedder":
            st.markdown(
                "**Average hits found by each embedder model across all datasets:**",
                help="Measure of power of different embedder models.",
            )
            st.markdown(
                "*Note: Random surrogate models are excluded from the comparison.*"
            )
        case "dataset":
            st.markdown(
                "**Average hits found in each dataset across all models and embedders:**",
                help="Measure of difficulty of the dataset for active learning simulations.",
            )

    _plot_cross_dataset_data(cross_dataset_data_average, "Average Hits")

    ### UNIQUE
    def unique_dict_constructor(exps):
        unique_hits, n_iteration_hits_std = _extract_unique_hits(exps)
        return {"unique_hits": len(unique_hits), "std": n_iteration_hits_std}

    cross_dataset_data_unique = _extract_cross_dataset_data(
        group_by, ds_groups, unique_dict_constructor
    )

    match group_by:
        case "model":
            st.markdown(
                "**Unique hits found by each surrogate model across all datasets:**",
                help="Measure of diversity of hits covered by different surrogate models.",
            )
            st.markdown(
                "*Note: Random surrogate models are excluded from the comparison.*"
            )
        case "embedder":
            st.markdown(
                "**Unique hits found by each embedder model across all datasets:**",
                help="Measure of diversity of hits covered by different embedder models.",
            )
            st.markdown(
                "*Note: Random surrogate models are excluded from the comparison.*"
            )
        case "dataset":
            st.markdown(
                "**Unique hits found in each dataset across all models and embedders:**",
                help="Measure of difficulty of the dataset for active learning simulations.",
            )

    _plot_cross_dataset_data(cross_dataset_data_unique, "Unique Hits")


def main():
    # ---------------------------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------------------------
    st.sidebar.header("Simulation results")

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

    selected_run = st.sidebar.selectbox(
        "Select Run", sorted(all_compressed_data.keys())
    )

    compressed_data = all_compressed_data[selected_run]
    experiments = _filter_experiments(compressed_data.experiments)

    # Group by dataset
    ds_groups: dict[str, list[DashboardExperimentData]] = {}
    for exp in experiments:
        dataset_name = exp.dataset_id.name
        if dataset_name not in ds_groups:
            ds_groups[dataset_name] = []
        ds_groups[dataset_name].append(exp)

    st.sidebar.markdown("---")
    selected_dataset = st.sidebar.selectbox("Select Dataset", sorted(ds_groups.keys()))
    dataset_id = ALSimulatorDataset[selected_dataset]
    definition = dataset_id.definition()
    pool = _load_pool(selected_dataset)
    exps = ds_groups[selected_dataset]

    with st.sidebar.expander("Dataset", expanded=True):
        st.markdown(
            f"**Optimization Mode:** {definition.optimization_mode.value}",
            help=f"The goal of this simulation: {definition.explain_optimization_mode()}",
        )

        if definition.target_lb is not None:
            st.markdown(f"**Target Lower Bound:** {definition.target_lb}")
        if definition.target_ub is not None:
            st.markdown(f"**Target Upper Bound:** {definition.target_ub}")
        if definition.target_value is not None:
            st.markdown(f"**Target Value:** {definition.target_value}")
        if definition.discrete_targets:
            st.markdown(
                f"**Discrete Targets:** {', '.join(definition.discrete_targets)}"
            )

        selector = definition.pool_selector
        st.markdown(
            f"**Pool:** {len(pool):,} sequences",
            help=f"Restricted to {selector.describe()}."
            if selector
            else "Every sequence in the dataset.",
        )
        st.markdown(
            f"**Number of potential hits:** {_potential_hits_summary(exps[0], len(pool))}",
            help=_hit_definition(exps[0]),
        )

    subset = ds_groups[selected_dataset]

    main_tabs = st.tabs(
        ["Individual Simulation Result", "Comparison", "Dataset Statistics"]
    )

    with main_tabs[0]:  # Individual simulation result
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

    with main_tabs[1]:  # Inner simulation comparison (model/embedder)
        _render_inner_simulation_comparison(subset)

    with main_tabs[2]:  # Dataset statistics
        _render_dataset_tab(selected_dataset, exps)

    st.divider()

    # Render cross-dataset comparison
    _render_cross_dataset_comparison(ds_groups)


if __name__ == "__main__":
    main()
