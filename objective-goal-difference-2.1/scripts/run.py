"""Run Objective 2.1 (goal difference) on its own: data checks, the full analysis, the report and the README results.

    python scripts/run.py           # full run, also refreshes the results in this folder's README.md
    python scripts/run.py --quick   # fewer repeats, for a fast check (README left unchanged)

Run from this folder (objective-goal-difference-2.1/). From the project root use:
    python objective-goal-difference-2.1/scripts/run.py
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import goalDifferenceModel as model  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description="Objective 2.1 - goal difference (FIFA World Cup 2026, HIT140 Assessment 3)")
    parser.add_argument("--quick", action="store_true", help="fewer CV repeats, splits and simulations")
    args = parser.parse_args()
    if args.quick:
        model.quick_mode()

    start = time.time()
    print("Checking the dataset")
    model.run_checks()                      # steps 1-2: data wrangling + data checks

    results = model.run()                   # steps 3-11: data preparation through to robustness
    print("Report:", model.report_2_1(results))   # step 12: plain-English report

    if args.quick:
        print("\nREADME results left unchanged (quick runs use fewer repeats).")
    elif model.update_readme():
        print("\nREADME results updated.")
    print(f"\nFinished in {time.time() - start:.0f} seconds.")


if __name__ == "__main__":
    main()
