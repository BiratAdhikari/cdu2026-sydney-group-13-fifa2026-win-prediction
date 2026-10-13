"""Run the whole project: data checks, Objective 2.1, Objective 2.2, the reports and the README results.

    python main.py                  # full run of both objectives (about 4 minutes), also refreshes every README
    python main.py --quick          # fewer repeats, for a fast check
    python main.py --objective 2.2  # one objective only

Each objective also runs on its own with  python <objective folder>/scripts/run.py  (see README.md).
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
OBJECTIVE_2_1 = ROOT_DIR / "objective-goal-difference-2.1"
OBJECTIVE_2_2 = ROOT_DIR / "objective-goal-count-2.2"
README = ROOT_DIR / "README.md"

sys.path.insert(0, str(OBJECTIVE_2_1 / "models"))
sys.path.insert(0, str(OBJECTIVE_2_2 / "models"))
import goalCountModel as model_2_2  # noqa: E402
import goalDifferenceModel as model_2_1  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def update_readmes():
    """Rewrite the results sections of the project README and of both objective READMEs."""
    model_2_1.update_readme()
    model_2_2.update_readme()
    return model_2_1.write_readme_results(README, model_2_1.readme_sections() + model_2_2.readme_sections())


def main():
    parser = argparse.ArgumentParser(description="FIFA World Cup 2026 linear regression (HIT140 Assessment 3)")
    parser.add_argument("--quick", action="store_true", help="fewer CV repeats, splits and simulations")
    parser.add_argument("--objective", choices=["2.1", "2.2", "all"], default="all")
    args = parser.parse_args()
    if args.quick:
        model_2_1.quick_mode()
        model_2_2.quick_mode()

    start = time.time()
    print("Checking the datasets")
    model_2_1.run_checks()
    model_2_2.run_checks()

    if args.objective in ("2.1", "all"):
        results = model_2_1.run()
        print("Report:", model_2_1.report_2_1(results))
    if args.objective in ("2.2", "all"):
        results = model_2_2.run()
        print("Report:", model_2_2.report_2_2(results))

    if args.quick:
        print("\nREADME results left unchanged (quick runs use fewer repeats).")
    elif update_readmes():
        print("\nREADME results updated.")
    print(f"\nFinished in {time.time() - start:.0f} seconds.")


if __name__ == "__main__":
    main()
