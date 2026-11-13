from __future__ import annotations

from pydantic import BaseModel, Field

from .datasets import ALSimulatorDataset
from .splits import ALSimulatorSplit


class DashboardSingleSimulationData(BaseModel):
    label: str = Field(
        ...,
        description="Label identifying the simulation with seed, model type, and success status",
    )
    is_success: bool = Field(
        ...,
        description="Whether the simulation met the success criteria (e.g. reached target hits)",
    )
    seed: int = Field(..., description="Random seed used for this simulation run")
    stop_reasons: list[str] = Field(
        ...,
        description="Reasons why the simulation stopped (e.g., budget exhausted, convergence reached)",
    )
    iteration_metrics_total: list[float] = Field(
        ..., description="Mean metric values for all masked data points per iteration"
    )
    iteration_metrics_suggestions: list[float] = Field(
        ..., description="Mean metric values for suggested data points per iteration"
    )
    iteration_hits: list[list[str]] = Field(
        ..., description="Hits found in each iteration"
    )
    iteration_consecutive_failures: list[int] = Field(
        ..., description="Count of consecutive failures in each iteration"
    )
    iteration_results_count: int = Field(
        ..., description="Total number of iteration results in the simulation"
    )
    n_hits_threshold: int | None = Field(
        ...,
        description="Target number of successful hits required for convergence, "
        "None if the campaign only stops on its budget",
    )


class DashboardExperimentData(BaseModel):
    name: str = Field(..., description="Results file the experiment was read from")
    dataset_id: ALSimulatorDataset = Field(
        ..., description="Dataset the simulations ran on"
    )
    embedder: str = Field(..., description="Embedder name")
    model: str = Field(..., description="Surrogate model")
    split_id: ALSimulatorSplit | None = Field(
        None,
        description="Dataset split (None for dashboard files written before splits existed)",
    )
    summary: dict = Field(..., description="Summary values for the dashboard header")
    aggregated_hits: int = Field(..., description="Hits summed over all simulations")
    aggregated_suggestions: int = Field(
        ..., description="Suggestions summed over all simulations"
    )
    potential_hits: list[str] = Field(
        ..., description="Ids of the landscape members that count as hits"
    )
    per_sim_n_hits: list[list[int]] = Field(
        ..., description="Number of hits per iteration (one list per simulation)"
    )
    per_sim_metrics_total: list[list[float]] = Field(
        ...,
        description="Mean metric over all predicted data points per iteration, one list per simulation",
    )
    per_sim_metrics_suggestions: list[list[float]] = Field(
        ...,
        description="Mean metric over the suggested data points per iteration, one list per simulation",
    )
    per_sim_is_success: list[bool] = Field(
        ..., description="Whether each simulation met its success criteria"
    )
    single_sims: list[DashboardSingleSimulationData] = Field(
        ..., description="Per-simulation data for the drill-down view"
    )


class DashboardCompressedData(BaseModel):
    run_name: str = Field(
        ..., description="Name of the simulation run the experiments belong to"
    )
    experiments: list[DashboardExperimentData] = Field(
        ..., description="One entry per results file of the run"
    )
