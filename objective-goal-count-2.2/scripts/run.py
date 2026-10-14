"""Run Objective 2.2 (goals scored) on its own: data checks, the full analysis, the report and the README results.

    python scripts/run.py           # full run, also refreshes the results in this folder's README.md
    python scripts/run.py --quick   # fewer repeats, for a fast check (README left unchanged)

Run from this folder (objective-goal-count-2.2/). From the project root use:
    python objective-goal-count-2.2/scripts/run.py

The data checks read ../objective-goal-difference-2.1/data/raw.csv to confirm the goals match the 2.1 scores, and the
2.1 vs 2.2 comparison uses the Objective 2.1 test predictions, so run Objective 2.1 first (or use main.py).
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import goalCountModel as model 

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description="Objective 2.2 - goals scored (FIFA World Cup 2026, HIT140 Assessment 3)")
    parser.add_argument("--quick", action="store_true", help="fewer CV repeats, splits and simulations")
    args = parser.parse_args()
    if args.quick:
        model.quick_mode()

    start = time.time()
    print("Checking the dataset")
    model.run_checks()                      # steps 1-2: data wrangling + data checks

    results = model.run()                   # steps 3-11: data preparation through to the 2.1 vs 2.2 comparison
    print("Report:", model.report_2_2(results))   # step 12: plain-English report

    if args.quick:
        print("\nREADME results left unchanged (quick runs use fewer repeats).")
    elif model.update_readme():
        print("\nREADME results updated.")
    print(f"\nFinished in {time.time() - start:.0f} seconds.")


if __name__ == "__main__":
    main()
