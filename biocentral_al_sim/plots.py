from __future__ import annotations

import re

import altair as alt
import numpy as np
import pandas as pd


def _extract_consecutive_failures_threshold(
    stop_reasons: list[str] | None,
) -> int | None:
    if not stop_reasons:
        return None
    for reason in stop_reasons:
        if "consecutive failures" in reason.lower():
            match = re.search(r"\((\d+)\)", reason)
            if match:
                return int(match.group(1))
    return None


def _align_iteration_lengths(*lists: list | None) -> tuple[int, list[list]]:
    lengths = [len(lst or []) for lst in lists]
    n = max(lengths) if lengths else 0
    aligned = [(lst or [])[:n] for lst in lists]
    return n, aligned


def _compute_convergence_stats(
    per_sim_n_hits: list[list[int]],
    n_hits_threshold: int | None,
) -> tuple[list[int], float | None, float | None]:
    if n_hits_threshold is None:
        return [], None, None
    convergence_iterations: list[int] = []
    for successes in per_sim_n_hits:
        if not successes:
            continue
        cumsum = np.cumsum(successes)
        converged_at = np.where(cumsum >= n_hits_threshold)[0]
        if len(converged_at) > 0:
            convergence_iterations.append(int(converged_at[0]) + 1)
    if not convergence_iterations:
        return convergence_iterations, None, None
    return (
        convergence_iterations,
        float(np.mean(convergence_iterations)),
        float(np.std(convergence_iterations)),
    )


def build_metric_evolution_chart(
    iteration_metrics_total: list[float],
    iteration_metrics_suggestions: list[float],
    n_iteration_hits: list[int],
    metric_name: str,
) -> alt.LayerChart:
    n, (total, suggestions, successes) = _align_iteration_lengths(
        iteration_metrics_total,
        iteration_metrics_suggestions,
        n_iteration_hits,
    )
    iterations = list(range(1, n + 1))

    rows: list[dict] = []
    for i, v in enumerate(total):
        rows.append(
            {"iteration": iterations[i], "series": "All Data", "value": float(v)}
        )
    for i, v in enumerate(suggestions):
        rows.append(
            {"iteration": iterations[i], "series": "Suggestions", "value": float(v)}
        )
    df = pd.DataFrame(rows)

    line = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("iteration:Q", title="Iteration"),
            y=alt.Y("value:Q", title=f"Metric ({metric_name})"),
            color=alt.Color("series:N", title="Series"),
            tooltip=["iteration:Q", "series:N", alt.Tooltip("value:Q", format=".4f")],
        )
    )

    success_iters = [iterations[i] for i, s in enumerate(successes) if s > 0]
    success_values = [
        total[i] for i, s in enumerate(successes) if s > 0 and i < len(total)
    ]
    layers: list[alt.Chart] = [line]
    if success_iters and success_values:
        stars_df = pd.DataFrame(
            {"iteration": success_iters[: len(success_values)], "value": success_values}
        )
        stars = (
            alt.Chart(stars_df)
            .mark_point(
                shape="diamond",
                size=220,
                color="gold",
                filled=True,
                stroke="black",
                strokeWidth=1.2,
            )
            .encode(
                x="iteration:Q",
                y="value:Q",
                tooltip=[
                    alt.Tooltip("iteration:Q", title="Iteration"),
                    alt.Tooltip("value:Q", title="Metric", format=".4f"),
                ],
            )
        )
        layers.append(stars)

    return alt.layer(*layers).properties(
        title="Metric Comparison with Success Markers", height=320
    )


def build_n_hits_chart(n_iteration_hits: list[int]) -> alt.Chart:
    """Cumulative hits per round."""
    n_rounds = len(n_iteration_hits or [])
    df = pd.DataFrame(
        {
            "round": range(1, n_rounds + 1),
            "cumulative": np.cumsum(n_iteration_hits or []).astype(int),
        }
    )
    return (
        alt.Chart(df)
        .mark_line(point=True, color="#1b7837", strokeWidth=2)
        .encode(
            x=round_axis(n_rounds),
            y=alt.Y("cumulative:Q", title="Hits"),
            tooltip=[
                alt.Tooltip("round:Q", title="Round"),
                alt.Tooltip("cumulative:Q", title="Hits"),
            ],
        )
        .properties(title="Hits", height=320)
    )


def build_consecutive_failures_chart(
    iteration_consecutive_failures: list[int],
    stop_reasons: list[str] | None,
) -> alt.LayerChart:
    """Rounds in a row without a hit, at the end of each round."""
    failures = iteration_consecutive_failures or []
    df = pd.DataFrame({"round": range(1, len(failures) + 1), "failures": failures})

    step = (
        alt.Chart(df)
        .mark_line(interpolate="step-after", color="#d7301f", strokeWidth=2, point=True)
        .encode(
            x=round_axis(len(failures)),
            y=alt.Y("failures:Q", title="Consecutive failures"),
            tooltip=[
                alt.Tooltip("round:Q", title="Round"),
                alt.Tooltip("failures:Q", title="Consecutive failures"),
            ],
        )
    )
    layers: list[alt.Chart] = [step]

    threshold = _extract_consecutive_failures_threshold(stop_reasons)
    if threshold is not None:
        rule_df = pd.DataFrame(
            {"y": [threshold], "label": [f"Threshold ({threshold})"]}
        )
        rule = (
            alt.Chart(rule_df)
            .mark_rule(color="#990000", strokeDash=[6, 4])
            .encode(y="y:Q", tooltip=["label:N"])
        )
        layers.append(rule)

    return alt.layer(*layers).properties(title="Consecutive Failures", height=320)


def build_suggested_labels_chart(
    iteration_suggestions: list[list[str]],
    id2label: dict[str, str],
    unique_labels: list[str],
    optimization_targets: list[str] | None,
) -> alt.Chart:
    rows: list[dict] = []
    targets = set(optimization_targets or [])
    for iter_idx, sugg_ids in enumerate(iteration_suggestions, start=1):
        iter_labels = [id2label.get(sid, "<unknown>") for sid in sugg_ids]
        counts = {label: iter_labels.count(label) for label in unique_labels}
        for label, count in counts.items():
            display = f"★ {label}" if label in targets else label
            rows.append(
                {
                    "iteration": iter_idx,
                    "label": label,
                    "label_display": display,
                    "count": count,
                    "is_target": label in targets,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return (
            alt.Chart(pd.DataFrame({"iteration": [], "count": []}))
            .mark_bar()
            .properties(title="Distribution of Suggested Labels", height=320)
        )

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("iteration:O", title="Iteration"),
            y=alt.Y("count:Q", title="Suggested Labels (per iteration)", stack="zero"),
            color=alt.Color("label_display:N", title="Labels"),
            tooltip=[
                alt.Tooltip("iteration:O", title="Iteration"),
                alt.Tooltip("label_display:N", title="Label"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(title="Distribution of Suggested Labels", height=320)
    )
    return chart


def build_performance_summary_chart(
    success_count: int,
    total_runs: int,
    convergence_iterations: list[int],
    max_iterations: int,
) -> alt.Chart:
    success_rate = (success_count / total_runs * 100.0) if total_runs > 0 else 0.0
    failure_rate = 100.0 - success_rate

    if convergence_iterations:
        avg_conv = float(np.mean(convergence_iterations))
        std_conv = float(np.std(convergence_iterations))
        score = max(0.0, 100.0 * (1.0 - avg_conv / max(max_iterations, 1)))
        conv_label = f"Convergence Speed ({len(convergence_iterations)} of {total_runs} converged)"
        conv_annotation = f"Avg: {avg_conv:.1f} ± {std_conv:.1f} iter"
    else:
        score = 0.0
        conv_label = "Convergence Speed (none converged)"
        conv_annotation = "n/a"

    rows = [
        {
            "row": f"Success Rate ({success_count}/{total_runs} runs)",
            "segment": "Success/Fast",
            "value": success_rate,
            "annotation": f"{success_rate:.1f}%",
        },
        {
            "row": f"Success Rate ({success_count}/{total_runs} runs)",
            "segment": "Failure/Slow",
            "value": failure_rate,
            "annotation": f"{failure_rate:.1f}%",
        },
        {
            "row": conv_label,
            "segment": "Success/Fast",
            "value": score,
            "annotation": conv_annotation,
        },
        {
            "row": conv_label,
            "segment": "Failure/Slow",
            "value": 100.0 - score,
            "annotation": "",
        },
    ]
    df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=["Success/Fast", "Failure/Slow"],
        range=["#2ecc71", "#e74c3c"],
    )

    bars = (
        alt.Chart(df)
        .mark_bar(stroke="black", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "value:Q",
                stack="zero",
                title="Percentage / Score",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y("row:N", title=None, sort=None),
            color=alt.Color("segment:N", scale=color_scale, title="Segment"),
            tooltip=[
                alt.Tooltip("row:N", title="Metric"),
                alt.Tooltip("segment:N", title="Segment"),
                alt.Tooltip("value:Q", title="Value", format=".1f"),
                alt.Tooltip("annotation:N", title="Detail"),
            ],
        )
    )

    text_df = df[df["segment"] == "Success/Fast"].copy()
    text_df["x_mid"] = text_df["value"] / 2.0
    text = (
        alt.Chart(text_df)
        .mark_text(color="black", fontWeight="bold")
        .encode(
            x=alt.X("x_mid:Q"),
            y=alt.Y("row:N", sort=None),
            text="annotation:N",
        )
    )

    return alt.layer(bars, text).properties(
        title="Overall Performance Summary", height=180
    )


def build_mean_cumulative_successes_chart(
    per_sim_n_hits: list[list[int]],
    per_sim_is_success: list[bool],
    target_threshold: int | None,
) -> alt.LayerChart:
    """Cumulative hits per simulation and their mean. Without a target threshold there is no
    success to colour the simulations by, nor a target line."""
    if not per_sim_n_hits:
        return alt.layer(
            alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()
        ).properties(title="Target Discovery Progress", height=320)

    max_iter = max(len(s) for s in per_sim_n_hits)
    iterations = np.arange(1, max_iter + 1)

    padded = np.array(
        [list(s) + [0] * (max_iter - len(s)) for s in per_sim_n_hits],
        dtype=float,
    )
    cumulative = np.cumsum(padded, axis=1)
    mean_cum = cumulative.mean(axis=0)
    std_cum = cumulative.std(axis=0)

    indiv_rows = []
    for sim_idx, row in enumerate(cumulative):
        for it_idx, value in enumerate(row):
            indiv_rows.append(
                {
                    "iteration": int(iterations[it_idx]),
                    "value": float(value),
                    "sim": sim_idx,
                    "is_success": bool(per_sim_is_success[sim_idx])
                    if sim_idx < len(per_sim_is_success)
                    else False,
                }
            )
    indiv_df = pd.DataFrame(indiv_rows)

    summary_df = pd.DataFrame(
        {
            "iteration": iterations,
            "mean": mean_cum,
            "low": np.maximum(0, mean_cum - std_cum),
            "high": mean_cum + std_cum,
        }
    )

    tooltip = [
        alt.Tooltip("iteration:Q", title="Iteration"),
        alt.Tooltip("value:Q", title="Cumulative"),
        alt.Tooltip("sim:N", title="Sim #"),
    ]
    individual = (
        alt.Chart(indiv_df)
        .mark_line(opacity=0.25, strokeWidth=1, color="grey")
        .encode(
            x=alt.X("iteration:Q", title="Iteration"),
            y=alt.Y("value:Q", title="Cumulative Target Successes"),
            detail="sim:N",
            tooltip=tooltip,
        )
    )
    if target_threshold is not None:
        individual = individual.encode(
            color=alt.Color(
                "is_success:N",
                scale=alt.Scale(domain=[True, False], range=["#2ecc71", "#e74c3c"]),
                legend=alt.Legend(title="Run success"),
            ),
            tooltip=[*tooltip, alt.Tooltip("is_success:N", title="Success")],
        )

    band = (
        alt.Chart(summary_df)
        .mark_area(opacity=0.3, color="#16a085")
        .encode(
            x="iteration:Q",
            y="low:Q",
            y2="high:Q",
        )
    )
    mean_line = (
        alt.Chart(summary_df)
        .mark_line(point=True, strokeWidth=3, color="#16a085")
        .encode(
            x="iteration:Q",
            y=alt.Y("mean:Q", title="Cumulative Target Successes"),
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteration"),
                alt.Tooltip("mean:Q", title="Mean", format=".2f"),
                alt.Tooltip("low:Q", title="-1 SD", format=".2f"),
                alt.Tooltip("high:Q", title="+1 SD", format=".2f"),
            ],
        )
    )
    layers = [individual, band, mean_line]
    if target_threshold is not None:
        rule_df = pd.DataFrame(
            {"y": [target_threshold], "label": [f"Target ({target_threshold})"]}
        )
        layers.append(
            alt.Chart(rule_df)
            .mark_rule(color="red", strokeDash=[6, 4], strokeWidth=2)
            .encode(y="y:Q", tooltip=["label:N"])
        )

    return alt.layer(*layers).properties(title="Target Discovery Progress", height=360)


def build_mean_metric_evolution_chart(
    per_sim_metrics_total: list[list[float]],
    best_run_metrics: list[float],
    worst_run_metrics: list[float],
    metric_name: str,
) -> alt.LayerChart:
    if not per_sim_metrics_total:
        return alt.layer(
            alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()
        ).properties(title=f"Model Performance Evolution ({metric_name})", height=320)

    max_iter = max(len(m) for m in per_sim_metrics_total)
    iterations = np.arange(1, max_iter + 1)

    padded = np.array(
        [list(m) + [np.nan] * (max_iter - len(m)) for m in per_sim_metrics_total],
        dtype=float,
    )
    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)

    summary_df = pd.DataFrame(
        {
            "iteration": iterations,
            "mean": mean,
            "low": mean - std,
            "high": mean + std,
        }
    )

    band = (
        alt.Chart(summary_df)
        .mark_area(opacity=0.3, color="#3498db")
        .encode(x="iteration:Q", y="low:Q", y2="high:Q")
    )
    mean_line = (
        alt.Chart(summary_df)
        .mark_line(point=True, strokeWidth=3, color="#3498db")
        .encode(
            x=alt.X("iteration:Q", title="Iteration"),
            y=alt.Y("mean:Q", title=metric_name),
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteration"),
                alt.Tooltip("mean:Q", title="Mean", format=".4f"),
                alt.Tooltip("low:Q", title="-1 SD", format=".4f"),
                alt.Tooltip("high:Q", title="+1 SD", format=".4f"),
            ],
        )
    )

    overlay_rows: list[dict] = []
    for it_idx, value in enumerate(best_run_metrics or [], start=1):
        overlay_rows.append(
            {"iteration": it_idx, "value": float(value), "run": "Best Run"}
        )
    for it_idx, value in enumerate(worst_run_metrics or [], start=1):
        overlay_rows.append(
            {"iteration": it_idx, "value": float(value), "run": "Worst Run"}
        )
    overlay_df = pd.DataFrame(overlay_rows)

    overlay = (
        alt.Chart(overlay_df)
        .mark_line(strokeDash=[6, 4], point=True, opacity=0.7, strokeWidth=2)
        .encode(
            x="iteration:Q",
            y=alt.Y("value:Q", title=metric_name),
            color=alt.Color(
                "run:N",
                scale=alt.Scale(
                    domain=["Best Run", "Worst Run"], range=["#2ecc71", "#e74c3c"]
                ),
                title="Run",
            ),
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteration"),
                alt.Tooltip("run:N", title="Run"),
                alt.Tooltip("value:Q", title=metric_name, format=".4f"),
            ],
        )
    )

    return alt.layer(band, mean_line, overlay).properties(
        title=f"Model Performance Evolution ({metric_name})", height=360
    )


def round_axis(n_rounds: int, field: str = "round") -> alt.X:
    """x channel for campaign rounds: ticks and grid lines on whole rounds only."""
    return alt.X(
        f"{field}:Q",
        title="Round",
        scale=alt.Scale(domain=[0.5, max(n_rounds, 1) + 0.5], nice=False),
        axis=alt.Axis(values=list(range(1, n_rounds + 1)), format="d"),
    )


# Dashed reference lines (random model, perfect calibration), each with a legend entry
_DASHED = alt.StrokeDash(
    "series:N",
    title=None,
    scale=alt.Scale(range=[[6, 4]]),
    legend=alt.Legend(symbolType="stroke"),
)


_CAP_WIDTH = 10  # about half the bar width


def build_grouped_metric_bars(
    replicates: pd.DataFrame,
    baselines: pd.DataFrame,
    titles: dict[str, str],
    encoder_order: list[str],
    columns: int = 2,
) -> alt.ConcatChart:
    """One subplot per metric. Per encoder, a bar per surrogate model with the mean over its
    replicates, standard error whiskers and the replicates as dots. Random model as dashed line.

    Args:
        replicates: Columns metric, encoder, model, value, a row per replicate.
        baselines: The random model's replicates, same columns; may be empty.
        titles: Subplot title by metric, in display order.
        encoder_order: Encoders in x axis order.
        columns: Subplots per row.
    """
    summary = (
        replicates.groupby(["metric", "encoder", "model"])["value"]
        .agg(mean="mean", se="sem")
        .fillna({"se": 0.0})
        .reset_index()
    )
    summary["low"] = summary["mean"] - summary["se"]
    summary["high"] = summary["mean"] + summary["se"]
    x = alt.X(
        "encoder:N",
        title="Encoder",
        sort=encoder_order,
        # Streamlit's chart theme otherwise hides labels whose boxes overlap
        axis=alt.Axis(labelAngle=-30, labelLimit=220, labelOverlap=False),
    )
    offset = alt.XOffset("model:N")
    subplots = []
    for metric, title in titles.items():
        means = alt.Chart(summary[summary["metric"] == metric]).encode(
            x=x, xOffset=offset
        )
        layers = [
            means.mark_bar().encode(
                y=alt.Y("mean:Q", title=None),
                color=alt.Color("model:N", title="Surrogate model"),
                tooltip=[
                    alt.Tooltip("encoder:N", title="Encoder"),
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("mean:Q", title="Mean", format=".3g"),
                    alt.Tooltip("se:Q", title="Standard error", format=".3g"),
                ],
            ),
            # Drawn as a rule and two caps, each offset like the bars
            means.mark_rule(color="black").encode(
                y=alt.Y("low:Q", title=None), y2="high:Q"
            ),
            means.mark_tick(color="black", size=_CAP_WIDTH, thickness=1.5).encode(
                y="low:Q"
            ),
            means.mark_tick(color="black", size=_CAP_WIDTH, thickness=1.5).encode(
                y="high:Q"
            ),
            alt.Chart(replicates[replicates["metric"] == metric])
            .mark_circle(color="black", size=20, opacity=0.6)
            .encode(
                x=x,
                xOffset=offset,
                y="value:Q",
                tooltip=[
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("value:Q", title="Replicate", format=".3g"),
                ],
            ),
        ]
        baseline = baselines[baselines["metric"] == metric]
        if not baseline.empty:
            layers.append(
                alt.Chart(
                    pd.DataFrame(
                        {
                            "value": [baseline["value"].mean()],
                            "series": ["Random model"],
                        }
                    )
                )
                .mark_rule(color="grey", strokeWidth=2)
                .encode(
                    y="value:Q",
                    strokeDash=_DASHED,
                    tooltip=[
                        alt.Tooltip("value:Q", title="Random model", format=".3g")
                    ],
                )
            )
        subplots.append(
            alt.layer(*layers).properties(title=title, width=840 // columns, height=260)
        )
    return alt.concat(*subplots, columns=columns)


def build_replicate_rounds_chart(
    rounds: pd.DataFrame,
    y_title: str,
    mean_line: bool,
    target: float | None = None,
    log_y: bool = False,
) -> alt.LayerChart | alt.FacetChart:
    """Value per round of each replicate colored by seed. Panel column gives row of subplots per panel.

    Args:
        rounds: Columns seed, round, value and optionally panel.
        y_title: Title of the y axis.
        mean_line: Points per replicate and a line for their mean, instead of a line per
            replicate.
        target: Draw a dashed target line at this value; not with panels.
        log_y: Draw the y axis on a log scale.
    """
    seeds = list(dict.fromkeys(rounds["seed"]))
    n_rounds = int(rounds["round"].max()) if not rounds.empty else 1
    # keep replicates with equal values visible
    nudge = {seed: (k - (len(seeds) - 1) / 2) * 0.08 for k, seed in enumerate(seeds)}
    data = rounds.assign(
        x=rounds["round"] + rounds["seed"].map(nudge), seed=rounds["seed"].astype(str)
    )
    y_scale = alt.Scale(type="log") if log_y else alt.Undefined
    color = alt.Color("seed:N", title="Seed", sort=[str(seed) for seed in seeds])
    tooltip = [
        alt.Tooltip("seed:N", title="Seed"),
        alt.Tooltip("round:Q", title="Round"),
        alt.Tooltip("value:Q", title=y_title, format=".3g"),
    ]
    if mean_line:
        layers = [
            alt.Chart()
            .mark_point(filled=True, size=50)
            .encode(
                x=round_axis(n_rounds, field="x"),
                y=alt.Y("value:Q", title=y_title, scale=y_scale),
                color=color,
                tooltip=tooltip,
            ),
            alt.Chart()
            .mark_line(color="black")
            .encode(
                x=round_axis(n_rounds),
                y=alt.Y("mean(value):Q", title=y_title, scale=y_scale),
            ),
        ]
    else:
        layers = [
            alt.Chart()
            .mark_line(point=True)
            .encode(
                x=round_axis(n_rounds),
                y=alt.Y("value:Q", title=y_title, scale=y_scale),
                color=color,
                tooltip=tooltip,
            )
        ]
    if "panel" in rounds.columns:
        return (
            alt.layer(*layers, data=data)
            .properties(width=600, height=220)
            .facet(
                row=alt.Row(
                    "panel:N", title=None, sort=list(dict.fromkeys(rounds["panel"]))
                )
            )
        )
    if target is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"target": [target]}))
            .mark_rule(color="red", strokeDash=[6, 4], strokeWidth=2)
            .encode(y="target:Q")
        )
    return alt.layer(*layers, data=data).properties(height=320)


def build_rounds_by_model_chart(
    rounds: pd.DataFrame,
    baseline: pd.DataFrame,
    y_title: str,
    encoder_order: list[str],
    log_y: bool = False,
    zero_line: bool = False,
) -> alt.FacetChart:
    """A subplot per surrogate model, and per panel if `rounds` has a panel column, with a line
    per encoder and the random model as a dashed line in each.

    Args:
        rounds: Columns model, encoder, round, value and optionally panel.
        baseline: The random model's rows, same columns; may be empty.
        y_title: Title of the y axis.
        encoder_order: Encoders in legend order.
        log_y: Draw the y axis on a log scale.
        zero_line: Mark zero with a black line, for changes.
    """
    panels = ["panel"] if "panel" in rounds.columns else []
    n_rounds = int(rounds["round"].max()) if not rounds.empty else 1
    y_scale = alt.Scale(type="log") if log_y else alt.Undefined
    base = alt.Chart().encode(
        x=round_axis(n_rounds), y=alt.Y("value:Q", title=y_title, scale=y_scale)
    )
    layers = [
        base.transform_filter(alt.datum.series == "encoder")
        .mark_line(point=True)
        .encode(
            color=alt.Color("encoder:N", title="Encoder", sort=encoder_order),
            tooltip=[
                alt.Tooltip("encoder:N", title="Encoder"),
                alt.Tooltip("round:Q", title="Round"),
                alt.Tooltip("value:Q", title=y_title, format=".3g"),
            ],
        )
    ]
    data = rounds.assign(series="encoder")
    if not baseline.empty:
        mean = baseline.groupby([*panels, "round"], as_index=False)["value"].mean()
        # Repeated per surrogate model, so the baseline shows in every subplot
        data = pd.concat(
            [data]
            + [
                mean.assign(model=model, series="Random model")
                for model in rounds["model"].unique()
            ],
            ignore_index=True,
        )
        layers.append(
            base.transform_filter(alt.datum.series == "Random model")
            .mark_line(color="grey")
            .encode(
                strokeDash=_DASHED,
                tooltip=[
                    alt.Tooltip("round:Q", title="Round"),
                    alt.Tooltip("value:Q", title="Random model", format=".3g"),
                ],
            )
        )
    if zero_line:
        layers.append(alt.Chart().mark_rule(color="black").encode(y=alt.datum(0)))
    facets = {"column": alt.Column("model:N", title=None)}
    if panels:
        facets["row"] = alt.Row(
            "panel:N", title=None, sort=list(dict.fromkeys(rounds["panel"]))
        )
    return (
        alt.layer(*layers, data=data).properties(width=300, height=220).facet(**facets)
    )


def build_calibration_chart(
    curves: pd.DataFrame, encoder_order: list[str]
) -> alt.FacetChart:
    """Observed against expected interval coverage, a subplot per surrogate model with a line
    per encoder and the diagonal for theoretical perfect calibration.

    Args:
        curves: Columns model, encoder, expected, observed.
        encoder_order: Encoders in legend order.
    """
    diagonal = pd.DataFrame({"expected": [0.0, 1.0], "observed": [0.0, 1.0]})
    data = pd.concat(
        [curves.assign(series="encoder")]
        + [
            diagonal.assign(model=model, series="Perfect calibration")
            for model in curves["model"].unique()
        ],
        ignore_index=True,
    )
    scale, axis = alt.Scale(domain=[0, 1]), alt.Axis(format="%")
    base = alt.Chart().encode(
        x=alt.X("expected:Q", title="Expected coverage", scale=scale, axis=axis),
        y=alt.Y("observed:Q", title="Observed coverage", scale=scale, axis=axis),
    )
    lines = (
        base.transform_filter(alt.datum.series == "encoder")
        .mark_line(point=True)
        .encode(
            color=alt.Color("encoder:N", title="Encoder", sort=encoder_order),
            tooltip=[
                alt.Tooltip("encoder:N", title="Encoder"),
                alt.Tooltip("expected:Q", title="Expected", format=".0%"),
                alt.Tooltip("observed:Q", title="Observed", format=".1%"),
            ],
        )
    )
    reference = (
        base.transform_filter(alt.datum.series == "Perfect calibration")
        .mark_line(color="grey")
        .encode(strokeDash=_DASHED)
    )
    return (
        alt.layer(lines, reference, data=data)
        .properties(width=300, height=260)
        .facet(column=alt.Column("model:N", title=None))
    )


def _round_bin_edges(values: np.ndarray, max_bins: int = 30) -> np.ndarray:
    """Set histogram bin edges of multiples of 1, 2, 5 and a power of 10."""
    low, high = float(values.min()), float(values.max())
    if high == low:
        return np.array([low - 0.5, low + 0.5])
    magnitude = 10 ** np.floor(np.log10((high - low) / max_bins))
    width = next(
        step * magnitude
        for step in (1, 2, 5, 10)
        if step * magnitude * max_bins >= high - low
    )
    # Rounded first so floating-point noise doesn't add an empty bin at either end
    first = np.floor(np.round(low / width, 9))
    last = np.ceil(np.round(high / width, 9))
    decimals = max(0, int(-np.floor(np.log10(width))))
    return np.round(np.arange(first, last + 1) * width, decimals)


def build_label_distribution_chart(
    labels: list[str],
    threshold: float | None,
    maximize: bool,
    label_names: dict[str, str] | None = None,
) -> alt.Chart | alt.LayerChart:
    """Histogram of measured labels, with a dashed vertical rule for the hit threshold.

    With `label_names`, each label is a category drawn as one bar in label order, and the rule
    sits on the edge between the hit categories and the rest.
    """
    if label_names is None:
        # Bin it here so we don't give every label to Vega
        values = np.array([float(label) for label in labels])
        counts, edges = np.histogram(values, bins=_round_bin_edges(values))
        df = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "count": counts})
        bars = (
            alt.Chart(df)
            .mark_bar(color="steelblue")
            .encode(
                x=alt.X("start:Q", bin="binned", title="Score"),
                x2="end:Q",
                y=alt.Y("count:Q", title="Sequences"),
                tooltip=[
                    alt.Tooltip("start:Q", title="From", format=".2f"),
                    alt.Tooltip("end:Q", title="To", format=".2f"),
                    alt.Tooltip("count:Q", title="Sequences"),
                ],
            )
        )
        rule_data, rule_x = {"threshold": [threshold]}, alt.X("threshold:Q")
    else:
        try:
            order = sorted(label_names, key=float)
        except ValueError:
            order = sorted(label_names)
        names = [label_names[label] for label in order]
        counts = pd.Series(labels).value_counts()
        df = pd.DataFrame(
            {"name": names, "count": [int(counts.get(label, 0)) for label in order]}
        )
        bars = (
            alt.Chart(df)
            .mark_bar(color="steelblue")
            .encode(
                x=alt.X(
                    "name:N", sort=names, title="Score", axis=alt.Axis(labelAngle=0)
                ),
                y=alt.Y("count:Q", title="Sequences"),
                tooltip=[
                    alt.Tooltip("name:N", title="Label"),
                    alt.Tooltip("count:Q", title="Sequences"),
                ],
            )
        )
        if threshold is not None:
            hits = [
                label
                for label in order
                if (
                    float(label) >= threshold if maximize else float(label) <= threshold
                )
            ]
            edge = hits[0] if maximize else hits[-1]
            rule_data = {"name": [label_names[edge]]}
            rule_x = alt.X("name:N", sort=names, bandPosition=0 if maximize else 1)
    if threshold is None:
        return bars.properties(height=320)
    rule = (
        alt.Chart(pd.DataFrame(rule_data))
        .mark_rule(color="#d62728", strokeDash=[6, 4], strokeWidth=2)
        .encode(x=rule_x)
    )
    return alt.layer(bars, rule).properties(height=320)


def build_mutation_depth_chart(depths: dict[str, list[int]]) -> alt.Chart:
    """Histogram of mutational depth for entire pool and hits."""
    rows = []
    for group, values in depths.items():
        counts = pd.Series(values).value_counts()
        rows.extend(
            {"group": group, "depth": int(depth), "count": int(count)}
            for depth, count in counts.items()
        )
    df = pd.DataFrame(rows)
    df["share"] = df["count"] / df.groupby("group")["count"].transform("sum")
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "depth:O",
                title="Mutational depth",
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset("group:N", sort=list(depths)),
            y=alt.Y("share:Q", title="Share of group", axis=alt.Axis(format="%")),
            color=alt.Color("group:N", sort=list(depths), title=None),
            tooltip=[
                alt.Tooltip("group:N", title="Group"),
                alt.Tooltip("depth:O", title="Mutational depth"),
                alt.Tooltip("count:Q", title="Variants"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=320)
    )
