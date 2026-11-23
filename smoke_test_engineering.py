"""Runs the runner's own `main()` against the live server, on SACAS9 only, with one replicate
of two 5-variant rounds, and every grid axis switched on.
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from biocentral_api import CommonEmbedder

import run_simulations_engineering as grid
from biocentral_al_sim.datasets import ALSimulatorDataset
from biocentral_al_sim.results import ActiveLearningMultipleSimulationResult
from biocentral_al_sim.settings import CampaignSettings

logger = logging.getLogger("smoke_test_engineering")

SMOKE = CampaignSettings(
    short_name="smoke",
    n_start=10,
    n_suggestions_per_iteration=5,
    n_iterations=2,
    n_replicates=1,
)
EMBEDDERS = [CommonEmbedder.ONE_HOT_ENCODING.value, CommonEmbedder.ESM2_8M.value]
N_CANDIDATES = 100


def run(results_dir: Path, jobs: int) -> int:
    # main() reads these module globals at call time.
    grid.EMBEDDERS = EMBEDDERS
    grid.RESULTS_DIR = results_dir
    variants = [
        grid.engineering_settings(SMOKE, delta, generated, 1).model_copy(
            update={"n_candidates_per_iteration": N_CANDIDATES}
        )
        for delta in (False, True)
        for generated in (False, True)
    ]

    status = grid.main(
        variants, datasets=[ALSimulatorDataset.COMBINGYM_SACAS9], jobs=jobs
    )

    files = sorted(results_dir.glob("al_eng_*.json"))
    # Per candidate source: onehot x 3 models, esm2-8M x 2 learned models, x 2 deltas.
    expected = 2 * (3 + 2 + 2)
    problems = []
    if len(files) != expected:
        problems.append(f"expected {expected} result files, found {len(files)}")
    budget = SMOKE.max_labels_budget
    for path in files:
        result = ActiveLearningMultipleSimulationResult.from_json(path)
        replicates = result.simulation_results
        if len(replicates) != SMOKE.n_replicates:
            problems.append(f"{path.name}: {len(replicates)} replicates")
        for replicate in replicates:
            measured = replicate.n_measured()
            if not 0 < measured <= budget:
                problems.append(f"{path.name}: measured {measured} of {budget}")

    for problem in problems:
        logger.error(problem)
    if status or problems:
        logger.error("Smoke test FAILED; results kept in %s", results_dir)
        return 1
    logger.info("Smoke test passed: %d experiments in %s", len(files), results_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Where to write results; default a fresh temporary directory",
    )
    parser.add_argument("--jobs", type=int, default=2, metavar="N")
    args = parser.parse_args()
    results_dir = args.results_dir or Path(tempfile.mkdtemp(prefix="al_smoke_"))
    sys.exit(run(results_dir, args.jobs))
