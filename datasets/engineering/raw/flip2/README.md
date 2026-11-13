# FLIP2 Data

Protein fitness landscape splits from [FLIP2](https://flip.protein.properties) (Didi et al., ICML
2026). Two of its seven datasets are downloaded here:
**TrpB** (enzyme fitness, continuous) and **NucB** (nuclease activity, ordinal bins).

Splits are static files on GitHub Pages.
## Downloading

```bash
cd datasets/raw/flip2
uv run python download_flip2.py                         # trpb + nucb, all their splits
uv run python download_flip2.py --list                  # all 7 datasets / 16 splits
uv run python download_flip2.py --datasets ired pdz3    # fetch others
uv run python download_flip2.py --datasets trpb --splits by_position
```

## Contents

| Dataset | Splits | Rows | Seq length | Target | License |
| --- | --- | --- | --- | --- | --- |
| `trpb` | `one_to_many`, `two_to_many`, `by_position` | 228 298 | 389 | continuous fitness, −0.801 – 2.451 | CC0 |
| `nucb` | `two_to_many` | 55 759 | 142 | ordinal activity bin, 0 – 3 | CC-BY 4.0 |

* **TrpB** — β-subunit of tryptophan synthase; growth-based fitness over combinatorially complete
  landscapes. Exactly **20 variable positions** (104–108, 117–119, 162, 166, 182–186, 227, 228, 230,
  231, 301), all 20 amino acids observed at each, forming ten sub-landscapes.
  Johnston et al. 2024, *PNAS* 121(32):e2400439121.
* **NucB** — *Bacillus licheniformis* nuclease B activity at pH 7, from a four-round ML-guided design
  campaign. 118 of 142 positions vary; variants carry 1–23 substitutions.
  Thomas et al. 2025, *Cell Systems* 16(3):101236.

## CSV format

Four columns, gzipped. Column order differs between the datasets.

| Column | Meaning |
| --- | --- |
| `sequence` | full variant sequence |
| `target` | measured fitness (continuous for TrpB, ordinal bin for NucB) |
| `set` | `train` or `test` |
| `validation` | bool; a flagged subset *of* `train`, not a third disjoint set |


Split sizes:

| File | train | test | validation |
| --- | --- | --- | --- |
| `trpb/one_to_many.csv.gz` | 380 | 227 918 | 76 |
| `trpb/two_to_many.csv.gz` | 10 791 | 217 507 | 2 158 |
| `trpb/by_position.csv.gz` | 39 446 | 188 852 | 7 889 |
| `nucb/two_to_many.csv.gz` | 7 455 | 48 304 | 5 367 |

> **Warning:** `nucb/two_to_many` is not the mutation-count split its name and FLIP2's description
> ("train on 0, 1 or 2 mutations") imply — 4 326 of its 7 455 training variants carry more than two
> substitutions and 9 936 test variants carry two or fewer.

### NucB target encoding

`target` is an ordinal scale, not a raw measurement.

| `target` | Upstream `activity_level` | Rows | Share |
| --- | --- | --- | --- |
| 0 | `non-functional` | 33 890 | 60.8 % |
| 1 | `activity_greater_than_0` | 11 098 | 19.9 % |
| 2 | `activity_greater_than_WT` | 10 572 | 19.0 % |
| 3 | `activity_greater_than_A73R` | 199 | 0.36 % |

(A73R is a known improved single variant)

## Compiling

```bash
uv run datasets/engineering/raw/flip2/compile_flip2.py
uv run datasets/dataset_preprocessor.py
```

[compile_flip2.py](compile_flip2.py) writes `engineering/raw/flip2_<name>.fasta`; the shared
preprocessor then writes `engineering/flip2_<name>_max2000.fasta`.

**Splits are dropped.** The `set` and `validation` columns are discarded and every row kept.
For TrpB that means only one of the three split files is read, since
all three carry identical `sequence` and `target` columns row for row.

### Resulting FASTA Files

| FASTA | Sequences | Length | Label |
| --- | --- | --- | --- |
| `flip2_TrpB_max2000.fasta` | 228 298 | 389 | continuous, −0.801 – 2.451 |
| `flip2_NucB_max2000.fasta` | 55 760 | 142 | ordinal bins `0`–`3` |

### Sequence Ids and the Parent Sequence

FLIP2 contains no parent sequence for either dataset, so both are manually included.

| Dataset | Parent | Ids | Why |
| --- | --- | --- | --- |
| `TrpB` | **Tm9D8∗**, written to `trpb/TrpB_wt.fasta` | substitutions, `TrpB_E105A:T106A:G107A` | ≤ 4 substitutions per variant, so ids reach only 28 characters |
| `NucB` | wild-type NucB, written to `nucb/NucB_wt.fasta` and added to the pool as `NucB_WT` | source row index, `NucB_00000` | up to 23 substitutions per variant, which would run to 120 characters |

**TrpB's parent is Tm9D8∗**, an engineered *Thermotoga maritima* TrpB variant, **not** wild-type TmTrpB.
Johnston et al. 2024: *"Tm9D8∗ differs from wildtype TmTrpB by ten amino acid substitutions (P19G, E30G,
I69V, K96L, P140L, N167D, I184F, L213P, G228S, and T292S)."

**NucB's parent** comes from the [upstream record](https://storage.googleapis.com/nuclease_design/processed_data/landscape.csv)
(`num_mutations == 0`).


## Citation and License

FLIP2 redistributes all data under CC-BY 4.0, per the paper and its
[Zenodo record](https://doi.org/10.5281/zenodo.18433203). The `LICENSE` file served next to the split
files is still the peer-review-only license from submission, which terminates on publication. The
code in [J-SNACKKB/FLIP](https://github.com/J-SNACKKB/FLIP) is under the Academic Free License 3.0.

> Didi K, Alamdari S, Lu AX, Wittmann BJ, Johnston KE, Amini AP, Madani A, Czeneszew M, Dallago C,
> Yang KK. FLIP2: Expanding Protein Fitness Landscape Benchmarks for Real-World Machine Learning
> Applications. *Proceedings of the 43rd International Conference on Machine Learning*, PMLR
> 306:24724–24777 (2026). [proceedings.mlr.press/v306/didi26a](https://proceedings.mlr.press/v306/didi26a.html)

```bibtex
@InProceedings{pmlr-v306-didi26a,
  title     = {{FLIP}2: Expanding Protein Fitness Landscape Benchmarks for Real-World Machine Learning Applications},
  author    = {Didi, Kieran and Alamdari, Sarah and Lu, Alex Xijie and Wittmann, Bruce James and Johnston, Kadina E and Amini, Ava P and Madani, Ali and Czeneszew, Maya and Dallago, Christian and Yang, Kevin K},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages     = {24724--24777},
  year      = {2026},
  volume    = {306},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v306/didi26a.html}
}
```

The underlying data keeps its upstream license, and its study should be cited alongside FLIP2:

| Dataset | Upstream data | License | Study |
| --- | --- | --- | --- |
| `trpb` | [CaltechDATA](https://doi.org/10.22002/h5rah-5z170) | CC0 1.0 | Johnston KE, Almhjell PJ, Watkins-Dulaney EJ, Liu G, Porter NJ, Yang J, Arnold FH. A combinatorially complete epistatic fitness landscape in an enzyme active site. *PNAS* 121(32):e2400439121 (2024). [doi:10.1073/pnas.2400439121](https://doi.org/10.1073/pnas.2400439121) |
| `nucb` | [google-deepmind/nuclease_design](https://github.com/google-deepmind/nuclease_design) | CC-BY 4.0 (code: Apache 2.0) | Thomas N, Belanger D, Xu C, Lee H, Hirano K, Iwai K, Polic V, Nyberg KD, Hoff KG, Frenz L, Emrich CA, Kim JW, Chavarha M, Ramanan A, Agresti JJ, Colwell LJ. Engineering highly active nuclease enzymes with machine learning and high-throughput screening. *Cell Systems* 16(3):101236 (2025). [doi:10.1016/j.cels.2025.101236](https://doi.org/10.1016/j.cels.2025.101236) |
