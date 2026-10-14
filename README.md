# FIFA World Cup 2026 – Linear Regression (HIT140 Assessment 3)

Two linear regression studies on the 104 matches of the 2026 World Cup, using only information that was
available before kick-off:

- **Objective 2.1** predicts the goal difference of a match (Team A goals minus Team B goals) – 104 rows.
  Folder: [`objective-goal-difference-2.1/`](objective-goal-difference-2.1/README.md)
- **Objective 2.2** predicts how many goals one team scores in a match – 208 rows (two per match).
  Folder: [`objective-goal-count-2.2/`](objective-goal-count-2.2/README.md)

Each objective is a self-contained project with its own data, model code, run script, images and README. One command
at the top level runs both (data checks, analysis, figures, tables, two plain-English reports and the results sections
of every README).

## Team

| Student | Student ID | Main responsibility | Main files |
|---|---|---|---|
| Hemanta Adhikari | s403355 | Objective 2.1 data preparation, EDA, preprocessing and regression modelling | `objective-goal-difference-2.1/` (`data/raw/raw.csv`, `models/goalDifferenceModel.py`, `images/`, `outputs/`) |
| Birat Adhikari | s403356 | Objective 2.2 data preparation, EDA, preprocessing and regression modelling | `objective-goal-count-2.2/` (`data/raw/raw.csv`, `models/goalCountModel.py`, `images/`, `outputs/`) |
| Muhammad Hussnain | s406259 | Regression experiments, model selection, assumption checking and diagnostics for both objectives | Steps 5–9 of both model files (preprocessing, model builders, evaluation, diagnostics, experiments) |
| John Karki | s403518 | Data loading and checking, evaluation, visualisation and performance analysis for both objectives | Steps 1–3, 7 and 10 of both model files (data wrangling, data checks, data preparation, evaluation, visualisation) |

Shared by everyone: `main.py`, the reporting step (step 12), the READMEs, testing, interpreting the results, the final
report and the presentation. Links to the final documents are in [`documents-link.txt`](documents-link.txt).

## Running it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Both objectives at once** (from the project root):

```bash
python main.py                     # full run of both objectives, about 4 minutes
python main.py --quick             # fewer repeats, under a minute
python main.py --objective 2.2     # one objective only
```

**Each objective on its own:**

```bash
cd objective-goal-difference-2.1
python scripts/run.py              # Objective 2.1, full run
python scripts/run.py --quick      # Objective 2.1, fast check

cd ../objective-goal-count-2.2
python scripts/run.py              # Objective 2.2, full run
python scripts/run.py --quick      # Objective 2.2, fast check
```

The scripts also work from the project root (`python objective-goal-difference-2.1/scripts/run.py`), because every
path is worked out from the file's own location.

Run Objective 2.1 before Objective 2.2 when running them separately: the 2.2 data checks compare its goals with the
2.1 scores, and the last 2.2 step compares both models on the same 42 test matches using the 2.1 test predictions
(it is skipped with a message if they are not there yet). `main.py` always runs them in the right order.

Results are reproducible: every split, fold and simulation uses `random_state = 0`. A full run rewrites the
**Results** sections of the READMEs from the latest numbers: `main.py` updates this README and both objective READMEs,
and `scripts/run.py` updates that objective's README. Quick runs leave the READMEs alone, because they use fewer
repeats.

## The pipeline (same steps in both objectives)

Each model file (`models/goalDifferenceModel.py`, `models/goalCountModel.py`) holds every step of its objective, in
the order the analysis runs, each under a clearly marked heading:

| Step | What it does | Output |
|---|---|---|
| Settings | paths, variable lists, run settings (CV folds, repeats, `random_state`) | – |
| 1. Data wrangling | load `data/raw.csv`, rename the target, convert `"25.0%"` to numbers, parse dates; rebuild the 2.2 prior averages | – |
| 2. Data checks | rows, variable count, missing values, goal difference = goals A − goals B, 2.2 goals match 2.1 scores, priors rebuild from earlier matches | `outputs/data_checks/data_verification.csv`, `data/processed/cleaned_dataset.csv` (2.1) |
| 3. Data preparation | 60/40 train/test split (`random_state = 0`); 2.2 splits whole matches | `data/processed/train.csv`, `data/processed/test.csv` |
| 4. Descriptive and inferential statistics | descriptive table, confidence intervals, t-tests, two-way tables, normality, non-linearity, VIF | `outputs/tables/01–04_*.csv`, `images/01–07_*.png` |
| 5. Preprocessing | winsorising, Yeo-Johnson + z-scores, backward elimination, log(1 + goals) target | `outputs/tables/05–07_*`, transformation and VIF figures |
| 6. Model builders | baseline, simple LR, multiple LR, backward elimination, Ridge, Lasso | – |
| 7. Evaluation | MAE, MSE, RMSE, normalised RMSE, R², bootstrap CIs, simulation, Brier score, permutation importance | `outputs/tables/09–14_*` |
| 8. Diagnostics | statsmodels OLS steps, assumption tests, influence | `outputs/tables/05_ols_step*.txt`, `12_*` |
| 9. Experiments | repeated 5-fold CV, corrected resampled t-test, model selection, 200 repeated splits | `outputs/tables/08_*`, `15_*` |
| 10. Visualisation | every figure | `images/` |
| 11. Analysis (`run()`) | runs steps 3–10 in order and saves the final model | `models/final_model.joblib`, `outputs/tables/results_summary.json` |
| 12. Reporting | plain-English report and README results | `outputs/report/objective_2_x_report.md`, README results |

## Data sources

Both datasets were compiled by our group from the three sources named in the assessment brief:

- **FIFA Official Website** – https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics
- **The Stats Don't Lie** – https://www.thestatsdontlie.com/football/world-cup-2026/
- **FBref** – https://fbref.com/en/

## The data

### Objective 2.1 – one row per match (`objective-goal-difference-2.1/data/raw.csv`)

Target: `Goal_Difference` = Team A goals − Team B goals. All eight explanatory variables are Team A minus Team B:

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

### Objective 2.2 – one row per team per match (`objective-goal-count-2.2/data/raw.csv`)

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

### Checking the data

Every run starts by checking the datasets and writes the results to each objective's
`outputs/data_checks/data_verification.csv`:

- 104 and 208 rows, exactly eight explanatory variables each, at most four shared, and no missing values;
- the goal difference equals Team A goals minus Team B goals in every match, and the 2.2 goals add up to the 2.1 scores;
- both 2.2 prior averages can be rebuilt from earlier results alone, on all 208 rows.

The odd-looking `Days_Rest_Diff = −70` in match 13 is a real value, not a typo: Uruguay had not played since 31 March
(76 days), while Saudi Arabia had played six days before. It is kept and handled by winsorising.

## Method

| Step | What we did | Unit material |
|---|---|---|
| Descriptive statistics | mean, median, mode, variance, IQR, skewness; histograms and box plots | Weeks 2, 7 |
| Confidence intervals | z-interval for a mean, normal-approximation interval for a proportion | Week 3 |
| Hypothesis tests | one-sample t-test; Welch two-sample t-tests | Week 4 |
| Visualisation | seaborn histograms, heatmap, pair plots, relationship plots | Week 7 |
| Linearity | scatter plots with straight line and LOWESS curve; x² term tests; Ramsey RESET | Weeks 8, 9 |
| 60/40 split | `train_test_split(test_size=0.4, random_state=0)`; 2.2 splits whole matches so both rows stay together | Weeks 8, 9 |
| Standardisation and normality | z-scores (`StandardScaler`); winsorising + Yeo-Johnson (`PowerTransformer`); Shapiro-Wilk and Q-Q plots before and after | Week 9 |
| Collinearity | correlation heatmap, VIF, dropping a collinear variable | Week 9 |
| statsmodels OLS | summaries for raw → z-score → transformed → collinearity fix → backward elimination | Week 9 |
| Metrics and baseline | MAE, MSE, RMSE, normalised RMSE, R²; mean-predictor baseline | Weeks 8, 9 |
| Conditional probability | two-way tables, P(result \| favourite) | Week 10 |
| Simulation | 10,000 simulated results per test match | Week 11 |

On top of that we compared competing models with repeated 5-fold cross-validation on the training set only, using a
corrected resampled t-test (Nadeau and Bengio), because CV folds overlap. Every fitted step sits inside a scikit-learn
pipeline, so nothing from the test set or a validation fold leaks into training. The models are:

- the mean baseline, an Elo-only simple regression and p-value backward elimination, kept as **reference models**;
- multiple regression with z-scores, then with winsorising + Yeo-Johnson, then without the collinear FIFA rank (2.1);
- Ridge and Lasso, with the penalty chosen by inner CV;
- for 2.2, the same models on a **log(1 + goals)** target, back-transformed with Duan's smearing correction.

**How the final model is chosen: accuracy first, then validity.** Start from the candidate with the lowest CV RMSE.
Among the candidates that are not significantly worse than it, take the most accurate one whose residuals pass every
assumption test (Shapiro-Wilk, Jarque-Bera, Breusch-Pagan and RESET). The test set is used once, at the end.

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

## Where to find things

```
.
├── main.py                              runs the checks, both objectives, the reports and every README's results
├── README.md                            this file
├── requirements.txt
├── documents-link.txt                   links to the final report and presentation
├── objective-goal-difference-2.1/       Objective 2.1 – goal difference
│   ├── README.md                        how to run it on its own, its steps and its results
│   ├── data/
│   │   ├── raw.csv                      the dataset as collected (never modified)
│   │   └── processed/                   cleaned_dataset.csv and the exact 60/40 train.csv / test.csv
│   ├── images/                          all figures (01_… to 19_…)
│   ├── models/
│   │   ├── goalDifferenceModel.py       every step of the 2.1 analysis (settings, steps 1–12)
│   │   └── final_model.joblib           the fitted final model (written by a run)
│   ├── outputs/                         tables/, report/, data_checks/ (written by a run)
│   └── scripts/
│       └── run.py                       runs Objective 2.1 on its own
└── objective-goal-count-2.2/            Objective 2.2 – goals scored by a team
    ├── README.md
    ├── data/
    │   ├── raw.csv
    │   └── processed/                   the exact 60/40 train.csv / test.csv (split by match)
    ├── images/                          all figures (01_… to 20_…)
    ├── models/
    │   ├── goalCountModel.py            every step of the 2.2 analysis (settings, steps 1–12)
    │   └── final_model.joblib
    ├── outputs/                         tables/, report/, data_checks/
    └── scripts/
        └── run.py                       runs Objective 2.2 on its own
```

Each objective writes about 20 numbered figures and 25–30 tables, in the order of the analysis. The statsmodels
summaries are in `outputs/tables/05_ols_step*.txt` and `outputs/tables/12_final_model_ols_summary.txt`, and
`outputs/report/objective_2_x_report.md` explains the results figure by figure in plain English.

## Limitations

- 42 test matches is a small test set, so every test-set number has a wide bootstrap interval.
- Goals are rare, random events, which caps how well any pre-match model can do.
- Line-ups, injuries, red cards and tactics are not in the data.
- In 2.2 the host effect rests on 13 rows, and the prior averages carry little information in the first round.
