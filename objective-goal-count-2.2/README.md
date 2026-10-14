# Objective 2.2 – Predicting how many goals a team scores in a World Cup 2026 match

Part of the FIFA World Cup 2026 linear regression project (HIT140 Assessment 3) – see the
[project README](../README.md) for both objectives together.

We predict how many goals one team scores in a match from eight variables known before kick-off. One row per team per
match: 208 rows, two for each of the 104 matches.

**Main responsibility:** Birat Adhikari (s403356) – data preparation, EDA, preprocessing and regression modelling.

## Running it on its own

From this folder (`objective-goal-count-2.2/`), after `pip install -r ../requirements.txt`:

```bash
python scripts/run.py              # full run, about 2 minutes, also refreshes the Results below
python scripts/run.py --quick      # fewer repeats, for a fast check (README left unchanged)
```

From the project root the same thing is `python objective-goal-count-2.2/scripts/run.py`, and `python main.py` runs
Objective 2.1 and this objective together.

**Run Objective 2.1 first** when running this objective on its own. The data checks read
`../objective-goal-difference-2.1/data/raw.csv` to confirm that the goals here add up to the 2.1 scores, and the last
step compares both models on the same 42 test matches using the 2.1 test predictions
(`../objective-goal-difference-2.1/outputs/tables/10_test_predictions.csv`). Without them that comparison is skipped
with a message; everything else still runs.

## Folder contents

```
objective-goal-count-2.2/
├── README.md                     this file
├── data/
│   ├── raw/
│       ├── raw.csv                     the dataset as collected (never modified)
│   └── processed/
│       ├── train.csv             124 training rows = 62 whole matches (60%, step 3)
│       └── test.csv              84 test rows = the same 42 test matches as Objective 2.1 (40%, step 3)
├── images/                       all figures, numbered in the order of the analysis
├── models/
│   ├── goalCountModel.py         every step of the analysis
│   └── final_model.joblib        the fitted final model
├── outputs/
│   ├── data_checks/              data_verification.csv
│   ├── tables/                   all tables, statsmodels summaries and results_summary.json
│   └── report/                   objective_2_2_report.md, the results explained figure by figure
└── scripts/
    └── run.py                    runs everything for this objective
```

Everything except `data/raw.csv`, the code and this README is written by a run.

## The data (`data/raw.csv`)

Target: `Goals_Scored`. Four variables are shared with 2.1 (the maximum allowed), seen from the team's side
(team minus opponent); the other four are new:

| Variable | Meaning | Shared with 2.1 |
|---|---|:---:|
| `Elo_Rating_Diff` | Elo rating, team minus opponent | ✓ |
| `Past_30_Days_Win_Rate_Diff` | 30-day win rate, team minus opponent | ✓ |
| `Avg_Goals_Scored_L5_Diff` | Goals scored per game (last 5), team minus opponent | ✓ |
| `Avg_Goals_Conceded_L5_Diff` | Goals conceded per game (last 5), team minus opponent | ✓ |
| `Team_Prior_Avg_Goals_Scored` | Team's goals per game in its earlier matches at this World Cup | |
| `Opp_Prior_Avg_Goals_Conceded` | Opponent's goals conceded per game in its earlier matches at this World Cup | |
| `Knockout_Stage` | 1 for Round of 32 onwards | |
| `Host_Advantage` | 1 when the USA, Mexico or Canada play in their own country | |

The two prior averages are shrunk towards a typical World Cup scoring rate, so they exist from the first match:

```
prior average = (2 × 1.3 + goals in earlier World Cup matches) / (2 + number of earlier World Cup matches)
```

Every team starts at 1.3, and one big result cannot dominate. Only earlier matches are counted, so nothing leaks from
the match being predicted.

## The steps (`models/goalCountModel.py`)

| Step | What it does | Output |
|---|---|---|
| Settings | paths, the eight variables (six continuous, two 0/1), run settings (`random_state = 0`, grouped 5-fold CV × 20 repeats, 200 splits) | – |
| 1. Data wrangling | load `raw.csv`, parse dates; rebuild both prior averages from earlier World Cup matches only | – |
| 2. Data checks | 208 rows, 8 variables, at most 4 shared with 2.1, no missing values, every match twice, goals match the 2.1 scores, priors rebuild on all 208 rows | `outputs/data_checks/data_verification.csv` |
| 3. Data preparation | 60/40 split **by match** (`random_state = 0`), so both rows of a match stay on the same side | `data/processed/train.csv`, `test.csv` |
| 4. Descriptive and inferential statistics | descriptive table, CIs, Welch t-tests (favourites, hosts, knockout), Poisson dispersion test, P(goals \| Elo favourite) | `outputs/tables/01–03_*`, `images/01–03_*` |
| 4b. Correlation and linearity | heatmap, straight line vs LOWESS, x² tests, Ramsey RESET on goals vs log(1 + goals), pair plot | `outputs/tables/04_*`, `images/04–07_*` |
| 5. Preprocessing + statsmodels OLS steps | raw → z-score → winsorise + Yeo-Johnson → backward elimination → log(1 + goals) target; normality before/after, VIF | `outputs/tables/05–07_*`, `images/08–09_*` |
| 6–9. Competing models | baseline, Elo-only LR, multiple LR (raw, transformed, log target), backward elimination, Ridge, Lasso, Ridge + log target; repeated grouped 5-fold CV; accuracy-first, validity-second selection | `outputs/tables/08_*`, `images/10–11_*` |
| 7–8. Final model on the test set | MAE, MSE, RMSE, NRMSE, R², match-level bootstrap CIs, rounded accuracy, residual and assumption checks (with HC3 errors), raw vs log target, influence | `outputs/tables/09–12_*`, `images/12–15_*`, `models/final_model.joblib` |
| 7. Which variables matter | standardised coefficients, permutation importance, Lasso path | `outputs/tables/13–14_*`, `images/16–18_*` |
| 9. Robustness | 200 different 60/40 match splits | `outputs/tables/15_*`, `images/19_*` |
| 2.1 vs 2.2 | both models on the same 42 test matches, Poisson scoreline simulation and the noise floor | `outputs/tables/16–17_*`, `images/20_*` |
| 12. Reporting | plain-English report and the Results below | `outputs/report/objective_2_2_report.md`, `outputs/tables/results_summary.json` |

`run()` at the bottom of step 11 calls the steps in this order; `scripts/run.py` runs the data checks, `run()`, the
report and the README update.

<!-- results:start - written by python main.py or scripts/run.py, edits here are overwritten -->

## Results – Objective 2.2

**Selected model: Ridge + log(1 + goals)** (all eight variables, back-transformed with a smearing factor of 1.117).

| Model (84 test rows) | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Ridge + log(1 + goals) (selected)** | 0.926 | 1.188 | 0.175 |
| Ridge on raw goals | 0.959 | 1.212 | 0.141 |
| Backward elimination (reference) | 0.901 | 1.188 | 0.174 |
| Elo-only regression (reference) | 0.901 | 1.189 | 0.173 |
| Mean baseline | 1.081 | 1.321 | −0.021 |

- With raw goals as the target the residuals are not normal (Shapiro-Wilk
  p < 0.001, RESET p = 0.193). The selected
  model works on the log(1 + goals) scale, where its residuals pass every check: Shapiro-Wilk
  p = 0.882, Breusch-Pagan p = 0.339, RESET
  p = 0.837.
- Lasso had the lowest CV RMSE (1.234), but its residuals fail the Shapiro-Wilk and Jarque-Bera tests. Ridge + log(1 + goals) is statistically tied with Lasso (p = 0.569) and passes every check, so it was selected.
- Compared with the plain eight-variable regression, CV RMSE improved 1.4% and
  test R² went from 0.105 to 0.175.
  Over 200 random splits its R² also varies less than any other regression model we tried (SD 0.061).
  For 2.2 the transformations mainly fixed validity rather than raw accuracy.
- 83% of rounded predictions are within one goal.
- Elo favourites score 1.23 more goals than underdogs (p < 0.001). Host
  nations at home score 0.80 more (p = 0.035, only 13 rows). Teams score
  about the same number of goals in knockout
  games (p = 0.839). Goals are
  slightly over-dispersed compared with a Poisson count
  (variance/mean = 1.31, p = 0.002).
- Collinearity is not a problem (largest VIF 2.83).
  The skewed variables (team's prior goals scored and opponent's prior goals conceded) were fixed by winsorising and Yeo-Johnson.

## Objective 2.1 vs 2.2 on the same 42 matches

Both objectives hold out exactly the same 42 matches, so the two approaches can be compared directly:

| Route | RMSE | Winner correct | Brier score |
|---|---:|---:|---:|
| 2.1 direct (goal difference model) | 1.648 | 90% | 0.539 |
| 2.2 implied (team goals model) | 1.649 | 80% | 0.559 |
| average of both | 1.639 | 83% | 0.547 |

Two models built separately, on different datasets with different targets, land on almost the same error, which is a good sign that both are sound.

We also simulated scorelines from a Poisson distribution at the predicted rates. Even a model that knew every team's true
scoring rate would get a median RMSE of 1.25 for team goals and
1.76 for goal difference. Our models sit at 1.19 and
1.65, inside those ranges, so the modest R² values come mostly from the randomness of football, not from missing effort.

<!-- results:end -->

## Limitations

- Goals are rare, so even 208 rows hold a lot of noise.
- The host effect rests on 13 rows.
- The two prior averages are weak in the first round because every team starts at the same value.
- The model predicts expected goals, not exact scorelines.
