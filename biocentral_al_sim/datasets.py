from enum import Enum, auto
from pathlib import Path

from biocentral_api import ActiveLearningOptimizationMode
from biotrainer_core.data_classes import SequenceData
from biotrainer_core.input_files import read_FASTA
from pydantic import BaseModel, Field

from .splits import (
    ALSimulatorSplitDefinition,
    AnySelector,
    MutatedSites,
    MutationCount,
    SiteMatch,
)


class ALSimulatorDatasetDefinition(BaseModel):
    """Everything that defines a simulation dataset: its data, the pool its campaigns draw from
    and its optimization target"""

    fasta_path: str = Field(description="Path to the labelled simulation FASTA")
    optimization_mode: ActiveLearningOptimizationMode = Field(
        description="How the labels are optimized"
    )
    discrete_targets: list[str] | None = Field(
        default=None, description="Target labels (mode: DISCRETE)"
    )
    target_lb: float | None = Field(
        default=None, description="Lower bound of the target (mode: INTERVAL)"
    )
    target_ub: float | None = Field(
        default=None, description="Upper bound of the target (mode: INTERVAL)"
    )
    target_value: float | None = Field(
        default=None, description="Target value (mode: VALUE)"
    )
    reference_fasta_path: str | None = Field(
        default=None,
        description="Single-sequence FASTA with the wild type this dataset mutates. Only mutational "
        "landscapes have one; mutation-aware splits (see splits) require it.",
    )
    pool_selector: AnySelector | None = Field(
        default=None,
        description="Restricts the sequences campaigns draw from, None for every sequence. "
        "Mutation-aware selectors need reference_fasta_path.",
    )

    def explain_optimization_mode(self) -> str:
        match self.optimization_mode:
            case ActiveLearningOptimizationMode.MAXIMIZE:
                return "Maximize the target value."
            case ActiveLearningOptimizationMode.MINIMIZE:
                return "Minimize the target value."
            case ActiveLearningOptimizationMode.DISCRETE:
                return "Classify the sequences into discrete classes."
            case ActiveLearningOptimizationMode.INTERVAL:
                return "Find target sequences with labels in the specified interval."
            case ActiveLearningOptimizationMode.VALUE:
                return "Find targets with the specified value."
            case _:
                return "Unknown optimization mode."


class ALSimulatorDataset(Enum):
    # The values are stored in the compressed dashboard data, so do not reorder these members
    MELTOME_MAXIMIZE = auto()
    MELTOME_MINIMIZE = auto()
    SCL = auto()
    AMYLASE = auto()
    PHOT = auto()
    EXOTOX = auto()
    COMBINGYM_GB1 = auto()
    COMBINGYM_CREILOV = auto()
    COMBINGYM_CR9114 = auto()
    COMBINGYM_MTAGBFP2 = auto()
    COMBINGYM_SACAS9 = auto()
    FLIP2_TRPB = auto()
    FLIP2_NUCB = auto()

    @staticmethod
    def screening():
        """The datasets in the experiment grid: the screening pools."""
        return [
            ALSimulatorDataset.MELTOME_MAXIMIZE,
            ALSimulatorDataset.MELTOME_MINIMIZE,
            ALSimulatorDataset.SCL,
            ALSimulatorDataset.AMYLASE,
            ALSimulatorDataset.PHOT,
            ALSimulatorDataset.EXOTOX,
        ]

    @staticmethod
    def engineering():
        """Single-parent mutational landscapes. These declare a wild type, so they are the datasets
        the mutation-aware extrapolation splits in the splits module can be applied to."""
        return [
            ALSimulatorDataset.COMBINGYM_GB1,
            ALSimulatorDataset.COMBINGYM_CREILOV,
            ALSimulatorDataset.COMBINGYM_CR9114,
            ALSimulatorDataset.COMBINGYM_MTAGBFP2,
            ALSimulatorDataset.COMBINGYM_SACAS9,
            ALSimulatorDataset.FLIP2_TRPB,
            ALSimulatorDataset.FLIP2_NUCB,
        ]

    def definition(self) -> ALSimulatorDatasetDefinition:
        definition = _DATASET_DEFINITIONS.get(self.name)
        if definition is None:
            raise ValueError(f"No dataset definition for {self.name}.")
        return definition

    def reference_sequence(self) -> str | None:
        """Wild-type sequence of this dataset, or None if it is not a mutational landscape."""
        reference_path = self.definition().reference_fasta_path
        if reference_path is None:
            return None
        lines = Path(reference_path).read_text().splitlines()
        return "".join(
            line.strip() for line in lines if line and not line.startswith(">")
        )

    def to_path(self, path_override: dict[str, str] | None = None) -> str:
        path = None
        if path_override:
            path = path_override.get(self.name)
        if path is not None:
            return path
        return self.definition().fasta_path

    def pool(self) -> list[SequenceData]:
        """Sequences this dataset's campaigns draw from: its FASTA, restricted by pool_selector."""
        pool = read_FASTA(self.to_path())
        selector = self.definition().pool_selector
        if selector is None:
            return pool
        definition = ALSimulatorSplitDefinition(
            description=f"{self.name} restricted to {selector.describe()}.",
            pool_selector=selector,
        )
        resolved, _ = definition.resolve(
            pool, explicit_reference=self.reference_sequence()
        )
        return resolved


_DATASET_DEFINITIONS: dict[str, ALSimulatorDatasetDefinition] = {
    ALSimulatorDataset.MELTOME_MAXIMIZE.name: ALSimulatorDatasetDefinition(
        fasta_path="datasets/screening/biotrainer_meltome_mixed_max2000.fasta",
        optimization_mode=ActiveLearningOptimizationMode.MAXIMIZE,
    ),
    ALSimulatorDataset.MELTOME_MINIMIZE.name: ALSimulatorDatasetDefinition(
        fasta_path="datasets/screening/biotrainer_meltome_mixed_max2000.fasta",
        optimization_mode=ActiveLearningOptimizationMode.MINIMIZE,
    ),
    ALSimulatorDataset.SCL.name: ALSimulatorDatasetDefinition(
        fasta_path="datasets/screening/scl_max2000.fasta",
        optimization_mode=ActiveLearningOptimizationMode.DISCRETE,
        discrete_targets=["Peroxisome"],
    ),
    ALSimulatorDataset.AMYLASE.name: ALSimulatorDatasetDefinition(
        fasta_path="datasets/screening/amylase_pet_max2000.fasta",
        optimization_mode=ActiveLearningOptimizationMode.MAXIMIZE,
    ),
    ALSimulatorDataset.PHOT.name: ALSimulatorDatasetDefinition(
        fasta_path="datasets/screening/PHOT_CHLRE_Chen_2023_max2000.fasta",
        optimization_mode=ActiveLearningOptimizationMode.MAXIMIZE,
    ),
    ALSimulatorDataset.EXOTOX.name: ALSimulatorDatasetDefinition(
        fasta_path="datasets/screening/exotox_merged_max2000.fasta",
        optimization_mode=ActiveLearningOptimizationMode.DISCRETE,
        discrete_targets=["EXOTOXIN"],
    ),
}

_COMBINGYM_RAW = "datasets/engineering/raw/combingym"


def _combingym(name: str) -> ALSimulatorDatasetDefinition:
    """A CombinGym landscape: maximize the measured phenotype, wild type from the shipped FASTA.

    Every CombinGym phenotype here is "higher is better" (binding affinity, fluorescence, nuclease
    or nickase activity), so they are all MAXIMIZE. The label column each one uses is chosen in
    compile_combingym.py.
    """
    return ALSimulatorDatasetDefinition(
        fasta_path=f"datasets/engineering/combingym_{name}_max2000.fasta",
        reference_fasta_path=f"{_COMBINGYM_RAW}/{name}/{name}_wt.fasta",
        optimization_mode=ActiveLearningOptimizationMode.MAXIMIZE,
    )


_DATASET_DEFINITIONS.update(
    {
        ALSimulatorDataset.COMBINGYM_GB1.name: _combingym("GB1"),
        ALSimulatorDataset.COMBINGYM_CREILOV.name: _combingym("CreiLOV"),
        ALSimulatorDataset.COMBINGYM_CR9114.name: _combingym("CR9114"),
        ALSimulatorDataset.COMBINGYM_MTAGBFP2.name: _combingym("mTagBFP2"),
        ALSimulatorDataset.COMBINGYM_SACAS9.name: _combingym("SaCas9"),
    }
)


_FLIP2_RAW = "datasets/engineering/raw/flip2"

_DATASET_DEFINITIONS.update(
    {
        # Continuous growth-based fitness. The parent is Tm9D8*, an engineered TmTrpB variant rather than
        # wild-type TmTrpB, reconstructed by compile_flip2.py (see the FLIP2 README).
        ALSimulatorDataset.FLIP2_TRPB.name: ALSimulatorDatasetDefinition(
            fasta_path="datasets/engineering/flip2_TrpB_max2000.fasta",
            reference_fasta_path=f"{_FLIP2_RAW}/trpb/TrpB_wt.fasta",
            optimization_mode=ActiveLearningOptimizationMode.MAXIMIZE,
            # A union of 64 saturation libraries. Across the union of their positions a generated
            # double lands on a label 2.9% of the time; inside the dominant 4-site library, 99.7%.
            # TODO allow all sites, 2.9% is fine
            pool_selector=MutatedSites(
                positions=[183, 184, 227, 228], match=SiteMatch.ONLY
            ),
        ),
        # Ordinal ladder: 0 non-functional, 1 activity > 0, 2 activity > WT, 3 activity > A73R
        ALSimulatorDataset.FLIP2_NUCB.name: ALSimulatorDatasetDefinition(
            fasta_path="datasets/engineering/flip2_NucB_max2000.fasta",
            reference_fasta_path=f"{_FLIP2_RAW}/nucb/NucB_wt.fasta",
            optimization_mode=ActiveLearningOptimizationMode.MAXIMIZE,
            # No library structure at all -- 20_896 maximal position sets, the top 20 covering 1%
            # of the pool. Only its single substitutions are dense: 2_088 of 2_112 are labelled.
            pool_selector=MutationCount(max_mutations=1),
        ),
    }
)
