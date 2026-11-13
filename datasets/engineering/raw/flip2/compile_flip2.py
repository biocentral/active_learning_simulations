"""Turn the FLIP2 split CSVs into labelled FASTAs for `dataset_preprocessor.main()`.

Ignores splits and adds parent sequences.

Variant IDs use substitutions for TrpB since it has up to 4 substitutions per variant, for NucB the CSV row index is
used.

Usage (from this directory):

    uv run python compile_flip2.py                  # both datasets
    uv run python compile_flip2.py --datasets TrpB
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from biotrainer_core.data_classes import SequenceData
from biotrainer_core.input_files import write_FASTA

HERE = Path(__file__).parent
OUTPUT_DIR = (
    HERE.parent
)  # engineering/raw/, so the shared preprocessor writes into engineering/

# The NucB parent, from the upstream landscape's num_mutations == 0 row
NUCB_PARENT = (
    "MIKKWAVHLLFSALVLLGLSGGAAYSPQHAEGAARYDDVLYFPASRYPETGAHISDAIKAGHADVCTIERSGADKRRQ"
    "ESLKGIPTKPGFDRDEWPMAMCEEGGKGASVRYVSSSDNRGAGSWVGNRLNGYADGTRILFIVQ"
)


# The TrpB parent: Tm9D8*, an engineered TmTrpB variant, NOT wild-type TmTrpB. Johnston et al. 2024
# (PNAS 121(32):e2400439121): "Tm9D8* differs from wildtype TmTrpB by ten amino acid substitutions
# (P19G, E30G, I69V, K96L, P140L, N167D, I184F, L213P, G228S, and T292S)." Reconstructed from UniProt
# P50909 (TRPB1_THEMA, 389 aa) with those ten substitutions applied
TRPB_PARENT = (
    "MKGYFGPYGGQYVPEILMGALEELEAAYEGIMKDESFWKEFNDLLRDYAGRPTPLYFARRLSEKYGARVYLKREDLLH"
    "TGAHKINNAIGQVLLAKLMGKTRIIAETGAGQHGVATATAAALFGMECVIYMGEEDTIRQKLNVERMKLLGAKVVPVK"
    "SGSRTLKDAIDEALRDWITNLQTTYYVFGSVVGPHPYPIIVRNFQKVIGEETKKQIPEKEGRLPDYIVACVSGGSNAA"
    "GIFYPFIDSGVKLIGVEAGGEGLETGKHAASLLKGKIGYLHGSKTFVLQDDWGQVQVSHSVSAGLDYSGVGPEHAYWR"
    "ETGKVLYDAVTDEEALDAFIELSRLEGIIPALESSHALAYLKKINIKGKVVVVNLSGRGDKDLESVLNHPYVRERIR"
)


class Landscape(NamedTuple):
    source: str  # split CSV to read; any one of a dataset's splits carries the full landscape
    integer_labels: bool  # ordinal bins, so write "2" rather than "2.0"
    parent: str | None  # parent sequence, if one is available
    parent_label: str | None  # label to add the parent under, when the source omits it
    substitution_ids: (
        bool  # ids as substitutions vs the parent, else the source row index
    )
    note: str


LANDSCAPES: dict[str, Landscape] = {
    "TrpB": Landscape(
        source="trpb/one_to_many.csv.gz",
        integer_labels=False,
        parent=TRPB_PARENT,
        parent_label=None,  # the source already carries the parent
        substitution_ids=True,  # at most 4 substitutions per variant, so ids stay short
        note="Continuous growth-based fitness over ten combinatorially complete sub-landscapes "
        "across 20 positions. Any of the three split files gives the same landscape.",
    ),
    "NucB": Landscape(
        source="nucb/two_to_many.csv.gz",
        integer_labels=True,
        parent=NUCB_PARENT,
        # Omitted from the distributed CSV, so added back under the label its own upstream
        # row carries: activity_greater_than_0, which the target encoding maps to bin 1.
        parent_label="1",
        substitution_ids=False,  # up to 23 substitutions per variant would give unwieldy ids
        note="Ordinal activity ladder, not a measurement: 0 non-functional, 1 active, 2 better than "
        "wild type, 3 better than the known A73R variant (199 of 55 759 sequences).",
    ),
}


def _format_label(value, integer_labels: bool) -> str:
    return str(int(value)) if integer_labels else repr(float(value))


def compile_dataset(dataset: str) -> Path:
    landscape = LANDSCAPES[dataset]
    frame = pd.read_csv(HERE / landscape.source)

    for column in ("sequence", "target"):
        if column not in frame.columns:
            raise ValueError(
                f"{dataset}: {landscape.source} has no {column!r} column; "
                f"found {list(frame.columns)}"
            )

    # Splits are deliberately dropped -- see the module docstring.
    dropped_columns = [c for c in ("set", "validation") if c in frame.columns]

    if frame["target"].isna().any():
        raise ValueError(
            f"{dataset}: {int(frame['target'].isna().sum())} rows have no target; "
            f"FLIP2 is not expected to ship missing labels."
        )

    lengths = set(frame["sequence"].astype(str).str.len())
    if len(lengths) != 1:
        raise ValueError(
            f"{dataset}: sequence lengths {sorted(lengths)[:5]} are not uniform."
        )

    width = len(str(len(frame) - 1))
    parent = landscape.parent

    def make_id(index: int, sequence: str) -> str:
        if not landscape.substitution_ids:
            return f"{dataset}_{index:0{width}d}"
        changes = ":".join(
            f"{expected}{offset + 1}{actual}"
            for offset, (expected, actual) in enumerate(zip(parent, sequence))
            if expected != actual
        )
        return f"{dataset}_{changes or 'WT'}"

    sequence_data = [
        SequenceData(
            seq_id=make_id(index, str(sequence)),
            seq=str(sequence),
            label=_format_label(target, landscape.integer_labels),
        )
        for index, (sequence, target) in enumerate(
            zip(frame["sequence"], frame["target"])
        )
    ]

    if landscape.parent_label is not None:
        sequence_data.append(
            SequenceData(
                seq_id=f"{dataset}_WT", seq=parent, label=landscape.parent_label
            )
        )

    if len({data_point.seq_id for data_point in sequence_data}) != len(sequence_data):
        raise ValueError(f"{dataset}: sequence ids are not unique.")

    output_path = OUTPUT_DIR / f"flip2_{dataset}.fasta"
    n_written = write_FASTA(output_path, sequence_data)

    print(f"{dataset:<6} {n_written:>7,} sequences -> {output_path.name}")
    print(
        f"{'':<6} length={lengths.pop()}  labels={'int' if landscape.integer_labels else 'float'}"
        f"  dropped split columns={dropped_columns or 'none'}"
    )

    if landscape.parent is not None:
        _write_parent(dataset, sequence_data, landscape.parent)
    else:
        print(f"{'':<6} no parent sequence available (see module docstring)")
    return output_path


def _write_parent(dataset: str, pool: list[SequenceData], parent: str) -> None:
    """Write the parent as a single-record FASTA, after checking it against the pool."""
    if len(parent) != len(pool[0].seq):
        raise ValueError(
            f"{dataset}: parent length {len(parent)} != variant length {len(pool[0].seq)}."
        )

    distances = [
        sum(a != b for a, b in zip(parent, data_point.seq)) for data_point in pool
    ]
    n_exact = distances.count(0)
    if n_exact != 1:
        raise ValueError(
            f"{dataset}: expected the parent to appear exactly once among the variants, "
            f"found it {n_exact} time(s). Either the parent or the source is wrong."
        )
    # A mutagenesis library sits close to its parent; a wrong parent would put every variant far away.
    median = sorted(distances)[len(distances) // 2]
    if median > len(parent) / 4:
        raise ValueError(
            f"{dataset}: median distance to the parent is {median} of {len(parent)} "
            f"residues, far too high for a mutagenesis library. The parent looks wrong."
        )

    parent_path = HERE / dataset.lower() / f"{dataset}_wt.fasta"
    parent_path.write_text(f">{dataset}_wt\n{parent}\n")
    print(
        f"{'':<6} parent -> {parent_path.relative_to(HERE)}  "
        f"(present once, substitutions {min(distances)}-{max(distances)}, median {median})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(LANDSCAPES),
        help=f"Which landscapes to compile (default: {', '.join(sorted(LANDSCAPES))})",
    )
    args = parser.parse_args()

    unknown = [dataset for dataset in args.datasets if dataset not in LANDSCAPES]
    if unknown:
        print(
            f"Unknown dataset(s): {', '.join(unknown)}. Known: {', '.join(sorted(LANDSCAPES))}",
            file=sys.stderr,
        )
        return 1

    missing = [
        dataset
        for dataset in args.datasets
        if not (HERE / LANDSCAPES[dataset].source).exists()
    ]
    if missing:
        print(
            f"Missing source CSV for {', '.join(missing)}. Run download_flip2.py first.",
            file=sys.stderr,
        )
        return 1

    for dataset in args.datasets:
        compile_dataset(dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
