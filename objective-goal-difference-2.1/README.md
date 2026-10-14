# Objective 2.1 – Predicting the goal difference of World Cup 2026 matches

Part of the FIFA World Cup 2026 linear regression project (HIT140 Assessment 3) – see the
[project README](../README.md) for both objectives together.

We predict the goal difference of each of the 104 matches (Team A goals minus Team B goals) from eight variables known
before kick-off. One row per match.

**Main responsibility:** Hemanta Adhikari (s403355) – data preparation, EDA, preprocessing and regression modelling.

## Running it on its own

From this folder (`objective-goal-difference-2.1/`), after `pip install -r ../requirements.txt`:

```bash
python scripts/run.py              # full run, about 2 minutes, also refreshes the Results below
python scripts/run.py --quick      # fewer repeats, for a fast check (README left unchanged)
```

From the project root the same thing is `python objective-goal-difference-2.1/scripts/run.py`, and
`python main.py` runs this objective and Objective 2.2 together.

## Folder contents

```
objective-goal-difference-2.1/
├── README.md                     this file
├── data/
│   ├──  raw/
│       ├── raw.csv               the dataset as collected (never modified)
│   └── processed/
│       ├── cleaned_dataset.csv   target renamed, head-to-head % turned into numbers, dates parsed (step 2)
│       ├── train.csv             the 62 training matches (60%, step 3)
│       └── test.csv              the 42 test matches (40%, step 3)
├── images/                       all figures, numbered in the order of the analysis
├── models/
│   ├── goalDifferenceModel.py    every step of the analysis
│   └── final_model.joblib        the fitted final model
├── outputs/
│   ├── data_checks/              data_verification.csv
│   ├── tables/                   all tables, statsmodels summaries and results_summary.json
│   └── report/                   objective_2_1_report.md, the results explained figure by figure
└── scripts/
    └── run.py                    runs everything for this objective
```

Everything except `data/raw.csv`, the code and this README is written by a run.

## The data (`data/raw.csv`)

Target: `Goal_Difference` = Team A goals − Team B goals (the raw column is `Goal_Difference (Target)`). All eight
explanatory variables are Team A minus Team B:

| Variable | Meaning |
|---|---|
| `FIFA_Rank_Diff` | FIFA world ranking before the tournament |
| `Elo_Rating_Diff` | Elo rating before the match |
| `Past_30_Days_Win_Rate_Diff` | Share of games won in the previous 30 days |
| `Avg_Goals_Scored_L5_Diff` | Goals scored per game, last five internationals |
| `Avg_Goals_Conceded_L5_Diff` | Goals conceded per game, last five internationals |
| `Days_Rest_Diff` | Days since the previous match |
| `Avg_xG_Margin_Diff` | Average expected-goals margin before the match |
| `Head_To_Head_Win_Pct_Diff` | Historical head-to-head win percentage |

The odd-looking `Days_Rest_Diff = −70` in match 13 is a real value, not a typo: Uruguay had not played since 31 March
(76 days), while Saudi Arabia had played six days before. It is kept and handled by winsorising.

## The steps (`models/goalDifferenceModel.py`)

| Step | What it does | Output |
|---|---|---|
| Settings | paths, the eight variables, run settings (`random_state = 0`, 5-fold CV × 20 repeats, 200 splits) | – |
| 1. Data wrangling | load `raw.csv`, rename `Goal_Difference (Target)` → `Goal_Difference`, turn `"25.0%"` into `25.0`, parse dates | – |
| 2. Data checks | 104 rows, 8 variables, no missing values, goal difference = Team A goals − Team B goals in every match | `outputs/data_checks/data_verification.csv`, `data/processed/cleaned_dataset.csv` |
| 3. Data preparation | `train_test_split(test_size=0.4, random_state=0)`: 62 training and 42 test matches | `data/processed/train.csv`, `test.csv` |
| 4. Descriptive and inferential statistics | descriptive table, CI for the mean, one-sample and Welch t-tests, intercept test, proportion CI, two-way table and P(result \| Elo favourite) | `outputs/tables/01–03_*`, `images/01–03_*` |
| 4b. Correlation and linearity | heatmap, straight line vs LOWESS, x² tests, pair plot | `outputs/tables/04_*`, `images/04–06_*` |
| 5. Preprocessing + statsmodels OLS steps | raw → z-score → winsorise + Yeo-Johnson → drop collinear FIFA rank → backward elimination; normality before/after, VIF | `outputs/tables/05–07_*`, `images/07–08_*` |
| 6–9. Competing models | baseline, Elo-only LR, multiple LR (3 versions), backward elimination, Ridge, Lasso; repeated 5-fold CV on the training set; accuracy-first, validity-second selection | `outputs/tables/08_*`, `images/09–10_*` |
| 7–8. Final model on the test set | MAE, MSE, RMSE, NRMSE, R², bootstrap CIs, result accuracy, 10,000 simulations per match, Brier score, residual and assumption checks, influence | `outputs/tables/09–12_*`, `images/11–14_*`, `images/19_*`, `models/final_model.joblib` |
| 7. Which variables matter | standardised coefficients, permutation importance, Lasso path | `outputs/tables/13–14_*`, `images/15–17_*` |
| 9. Robustness | 200 different 60/40 splits | `outputs/tables/15_*`, `images/18_*` |
| 12. Reporting | plain-English report and the Results below | `outputs/report/objective_2_1_report.md`, `outputs/tables/results_summary.json` |

`run()` at the bottom of step 11 calls the steps in this order; `scripts/run.py` runs the data checks, `run()`, the
report and the README update.

<!-- results:start - written by python main.py or scripts/run.py, edits here are overwritten -->

## Results – Objective 2.1

**Selected model: Lasso** (all eight variables in, it keeps the FIFA rank, Elo rating and goals conceded (last 5) differences).

| Model (42 test matches) | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Lasso (selected)** | 1.283 | 1.648 | 0.313 |
| Backward elimination (reference) | 1.258 | 1.645 | 0.316 |
| Elo-only regression (reference) | 1.282 | 1.669 | 0.296 |
| Mean baseline | 1.478 | 1.993 | −0.004 |

- Cross-validated RMSE fell from 1.826 for the raw eight-variable regression to
  1.517 (17% lower).
- The model picked the winner in **27 of 30** test matches that had one.
  Its simulated win/draw/loss probabilities have a Brier score of 0.539, against
  0.651 for the baseline.
- All residual checks pass: Shapiro-Wilk p = 0.578,
  Breusch-Pagan p = 0.370, RESET p = 0.857,
  Durbin-Watson 1.83.
- The raw OLS has no significant coefficient, because FIFA rank and Elo are nearly the same signal (r = −0.96,
  VIF 22). Dropping FIFA rank brings every VIF under 3.4, and
  backward elimination ends with the Elo rating difference alone.
- **Elo carries most of the signal.** The Elo-only regression is statistically tied with Lasso (p = 0.327). Over 200 random 60/40
  splits its mean test R² is 0.374, against
  0.331 for Lasso. The raw eight-variable regression averages
  −0.04 with a standard deviation of 0.83,
  which shows why the transformations and penalties matter.
- Team A wins by 0.48 goals on average (95% CI 0.09–0.87,
  one-sample t-test p = 0.009), but once Elo is in the model the advantage
  disappears (p = 0.235): Team A is usually
  the stronger team. The higher-Elo team won 84% of matches that had a winner
  (95% CI 76–92%), and knockout games were
  not significantly closer (p = 0.098).

<!-- results:end -->

## Limitations

- 42 test matches is a small test set, so every test-set number has a wide bootstrap interval.
- Goals are rare, random events, which caps how well any pre-match model can do.
- Line-ups, injuries, red cards and tactics are not in the data.
- Team A is usually the stronger side, which the intercept test accounts for.
