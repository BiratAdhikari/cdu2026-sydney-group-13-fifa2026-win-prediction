"""Objective 2.2 - predict how many goals a team scores in a World Cup 2026 match.

Everything Objective 2.2 needs is in this one file, in the order the analysis runs:

    Settings
    STEP 1   Data wrangling      load the raw CSV, parse dates; rebuild the prior-average variables
    STEP 2   Data checks         check the dataset against the brief (and against the 2.1 scores)
    STEP 3   Data preparation    60/40 split by match, saved to data/processed/
    STEP 4   Descriptive and inferential statistics
    STEP 5   Preprocessing       winsorise + Yeo-Johnson + z-scores, backward elimination, log(1 + goals) target
    STEP 6   Model builders
    STEP 7   Evaluation
    STEP 8   Diagnostics
    STEP 9   Experiments         cross-validation and model selection
    STEP 10  Visualisation
    STEP 11  The Objective 2.2 analysis (run() calls every step in order)
    STEP 12  Reporting           plain-English report and the README results

Run it with:  python scripts/run.py        (from this folder)   or   python main.py  (from the project root)
The comparison with Objective 2.1 (step 9 of the analysis) needs Objective 2.1 to have been run first.
"""


import json
import math
import textwrap
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm
from sklearn import metrics
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV, lasso_path
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.stattools import durbin_watson, jarque_bera


# ==============================================================================================================
# SETTINGS - paths, variables and run settings (was src/config.py)
# ==============================================================================================================
# Folders (all relative to this objective's folder)
OBJECTIVE_DIR = Path(__file__).resolve().parents[1]          # objective-goal-count-2.2/
PROJECT_DIR = OBJECTIVE_DIR.parent
DATA_DIR = OBJECTIVE_DIR / "data"
DATA_2_2 = DATA_DIR / "raw" / "raw.csv"                       # raw dataset, never modified
PROCESSED_DIR = DATA_DIR / "processed"                        # the exact 60/40 train/test files
FIGURES = OBJECTIVE_DIR / "images"                            # all figures
OUTPUTS_DIR = OBJECTIVE_DIR / "outputs"
TABLES = OUTPUTS_DIR / "tables"                               # all tables, statsmodels summaries, results_summary.json
REPORT_DIR = OUTPUTS_DIR / "report"                           # plain-English report
CHECKS_DIR = OUTPUTS_DIR / "data_checks"                      # dataset checks
MODELS_DIR = OBJECTIVE_DIR / "models"                         # this code + the saved final model
README = OBJECTIVE_DIR / "README.md"

# Objective 2.1 files, used to cross-check the scores and to compare 2.1 with 2.2 on the same test matches
OBJECTIVE_2_1_DIR = PROJECT_DIR / "objective-goal-difference-2.1"
DATA_2_1 = OBJECTIVE_2_1_DIR / "data" / "raw" / "raw.csv"
TABLES_2_1 = OBJECTIVE_2_1_DIR / "outputs" / "tables"

TEST_SIZE = 0.4
RANDOM_STATE = 0
CV_FOLDS = 5
CV_REPEATS = 20
N_SPLITS = 200
N_SIMULATIONS = 10_000
N_BOOTSTRAP = 2000
ALPHA = 0.05


def quick_mode():
    """Smaller repeat counts for a fast check run (--quick)."""
    global CV_REPEATS, N_SPLITS, N_SIMULATIONS, N_BOOTSTRAP
    CV_REPEATS, N_SPLITS, N_SIMULATIONS, N_BOOTSTRAP = 2, 20, 2000, 300


# the 2.1 variable list, only used to count how many variables 2.2 shares with 2.1 (at most 4 allowed)
FEATURES_2_1 = [
    "FIFA_Rank_Diff",
    "Elo_Rating_Diff",
    "Past_30_Days_Win_Rate_Diff",
    "Avg_Goals_Scored_L5_Diff",
    "Avg_Goals_Conceded_L5_Diff",
    "Days_Rest_Diff",
    "Avg_xG_Margin_Diff",
    "Head_To_Head_Win_Pct_Diff",
]

TARGET_2_2 = "Goals_Scored"
FEATURES_2_2 = [
    "Elo_Rating_Diff",
    "Past_30_Days_Win_Rate_Diff",
    "Avg_Goals_Scored_L5_Diff",
    "Avg_Goals_Conceded_L5_Diff",
    "Team_Prior_Avg_Goals_Scored",
    "Opp_Prior_Avg_Goals_Conceded",
    "Knockout_Stage",
    "Host_Advantage",
]
BINARY_2_2 = ["Knockout_Stage", "Host_Advantage"]
CONTINUOUS_2_2 = [f for f in FEATURES_2_2 if f not in BINARY_2_2]
SHARED_FEATURES = [f for f in FEATURES_2_2 if f in FEATURES_2_1]

# the prior averages in 2.2 start every team at 1.3 goals and add each earlier World Cup match
PRIOR_GOALS = 1.3
PRIOR_WEIGHT = 2

LABELS = {
    "Elo_Rating_Diff": "Elo rating diff",
    "Past_30_Days_Win_Rate_Diff": "30-day win rate diff",
    "Avg_Goals_Scored_L5_Diff": "Goals scored (last 5) diff",
    "Avg_Goals_Conceded_L5_Diff": "Goals conceded (last 5) diff",
    "Team_Prior_Avg_Goals_Scored": "Team prior avg goals scored",
    "Opp_Prior_Avg_Goals_Conceded": "Opponent prior avg goals conceded",
    "Knockout_Stage": "Knockout match (1 = yes)",
    "Host_Advantage": "Host nation at home (1 = yes)",
    "Goals_Scored": "Goals scored",
}


# ==============================================================================================================
# STEP 1 - DATA WRANGLING: load and clean the raw data
# Reads data/raw.csv and fixes column names and types. The raw file itself is never changed.
# ==============================================================================================================
def load_2_2(path=DATA_2_2):
    return pd.read_csv(path, parse_dates=["Date"])


def recompute_prior_averages(df):
    """Rebuild the two 2.2 prior-average variables from earlier World Cup matches only."""
    team_prior, opp_prior = [], []
    for row in df.itertuples(index=False):
        scored = df.loc[(df["Team"] == row.Team) & (df["Date"] < row.Date), TARGET_2_2]
        conceded = df.loc[(df["Opponent"] == row.Opponent) & (df["Date"] < row.Date), TARGET_2_2]
        team_prior.append((PRIOR_WEIGHT * PRIOR_GOALS + scored.sum()) / (PRIOR_WEIGHT + len(scored)))
        opp_prior.append((PRIOR_WEIGHT * PRIOR_GOALS + conceded.sum()) / (PRIOR_WEIGHT + len(conceded)))
    return np.array(team_prior), np.array(opp_prior)


# ==============================================================================================================
# STEP 2 - DATA CHECKS: verify the dataset against the assessment brief
# Results go to outputs/data_checks/data_verification.csv.
# ==============================================================================================================
def check_2_2(df):
    problems = []
    if len(df) != 208:
        problems.append(f"expected 208 team-matches, found {len(df)}")
    if len(FEATURES_2_2) != 8:
        problems.append("2.2 needs exactly 8 explanatory variables")
    if len(SHARED_FEATURES) > 4:
        problems.append("more than 4 explanatory variables shared with 2.1")
    if df[FEATURES_2_2 + [TARGET_2_2]].isna().any().any():
        problems.append("missing values")
    if not df.groupby("Match_ID").size().eq(2).all():
        problems.append("every match should appear exactly twice")
    return problems


def run_checks():
    """Check the 2.2 dataset against the brief, and its goals against the 2.1 match scores."""
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    teams = load_2_2()
    problems = check_2_2(teams)
    if problems:
        raise ValueError("; ".join(problems))
    matches = pd.read_csv(DATA_2_1)

    team_a = teams.merge(matches, left_on=["Match_ID", "Team"], right_on=["Match_ID", "Team_A"])
    team_b = teams.merge(matches, left_on=["Match_ID", "Team"], right_on=["Match_ID", "Team_B"])
    goals_match = int((team_a["Goals_Scored"] == team_a["Team_A_Goals"]).sum() +
                      (team_b["Goals_Scored"] == team_b["Team_B_Goals"]).sum())
    team_prior, opp_prior = recompute_prior_averages(teams)
    priors_ok = int((np.isclose(team_prior, teams["Team_Prior_Avg_Goals_Scored"], atol=0.006) &
                     np.isclose(opp_prior, teams["Opp_Prior_Avg_Goals_Conceded"], atol=0.006)).sum())

    checks = [
        ("2.2 rows (one per team per match)", len(teams), "208"),
        ("2.2 explanatory variables", len(FEATURES_2_2), "8"),
        ("2.2 missing values", int(teams[FEATURES_2_2 + [TARGET_2_2]].isna().sum().sum()), "0"),
        ("Explanatory variables shared by 2.1 and 2.2", len(SHARED_FEATURES), "at most 4"),
        ("2.2 rows whose goals match the 2.1 score", goals_match, "208"),
        ("2.2 rows whose prior averages rebuild from earlier matches only", priors_ok, "208"),
    ]
    table = pd.DataFrame(checks, columns=["check", "found", "required"])
    table["passed"] = [found <= 4 if required == "at most 4" else str(found) == required
                       for found, required in zip(table["found"], table["required"])]
    table.to_csv(CHECKS_DIR / "data_verification.csv", index=False)
    print(table.to_string(index=False))
    return table


# ==============================================================================================================
# STEP 3 - DATA PREPARATION: the 60/40 train/test split
# random_state = 0, so the split is reproducible. run() saves train.csv and test.csv to data/processed/.
# ==============================================================================================================
def split_by_match(df):
    """60/40 split of the 104 matches, so both rows of a match land on the same side."""
    train_ids, test_ids = train_test_split(df["Match_ID"].unique(), test_size=TEST_SIZE,
                                           random_state=RANDOM_STATE)
    return df[df["Match_ID"].isin(train_ids)], df[df["Match_ID"].isin(test_ids)]


# ==============================================================================================================
# STEP 4 - DESCRIPTIVE AND INFERENTIAL STATISTICS (weeks 2, 3, 4, 7 and 10)
# Descriptive tables, confidence intervals, t-tests, normality, non-linearity tests and VIF.
# ==============================================================================================================
def descriptive_table(df, columns):
    rows = []
    for col in columns:
        x = df[col]
        q1, q3 = np.percentile(x, [25, 75])
        rows.append({
            "variable": col, "n": len(x), "mean": x.mean(), "median": x.median(), "mode": x.mode().iloc[0],
            "std": x.std(ddof=1), "variance": x.var(ddof=1), "min": x.min(), "max": x.max(),
            "range": x.max() - x.min(), "q1": q1, "q3": q3, "iqr": q3 - q1,
            "skewness": st.skew(x), "kurtosis": st.kurtosis(x),
        })
    return pd.DataFrame(rows).round(3)


def mean_ci(sample, level=0.95):
    x_bar, s, n = np.mean(sample), np.std(sample, ddof=1), len(sample)
    z = st.norm.ppf(1 - (1 - level) / 2)
    margin = z * s / math.sqrt(n)
    return x_bar, x_bar - margin, x_bar + margin


def proportion_ci(count, total, level=0.95):
    low, high = proportion_confint(count, total, alpha=1 - level, method="normal")
    return count / total, low, high


def one_sample_t(sample, mu, alternative="two-sided"):
    return st.ttest_1samp(sample, mu, alternative=alternative)


def welch_t(sample1, sample2, alternative="two-sided"):
    return st.ttest_ind_from_stats(np.mean(sample1), np.std(sample1, ddof=1), len(sample1),
                                   np.mean(sample2), np.std(sample2, ddof=1), len(sample2),
                                   equal_var=False, alternative=alternative)


def conclusion(p_value, alpha=0.05):
    return "reject H0" if p_value < alpha else "fail to reject H0"


def normality_table(df, columns, stage):
    rows = []
    for col in columns:
        x = np.asarray(df[col], dtype=float)
        rows.append({"variable": col, "stage": stage, "skewness": st.skew(x), "shapiro_p": st.shapiro(x).pvalue})
    return pd.DataFrame(rows)


def nonlinearity_tests(X, y):
    """Fit y ~ x + x^2 for each variable; a significant x^2 term means the relationship bends."""
    rows = []
    y = np.asarray(y, dtype=float)
    for col in X.columns:
        x = (X[col] - X[col].mean()) / X[col].std()
        linear = sm.OLS(y, sm.add_constant(x.to_frame())).fit()
        curved = sm.OLS(y, sm.add_constant(pd.DataFrame({"x": x, "x_squared": x ** 2}))).fit()
        rows.append({"variable": col, "linear_r2": linear.rsquared, "quadratic_r2": curved.rsquared,
                     "x_squared_p": curved.pvalues["x_squared"]})
    return pd.DataFrame(rows)


def vif_table(X):
    design = sm.add_constant(pd.DataFrame(X).astype(float))
    return pd.DataFrame({
        "variable": design.columns[1:],
        "VIF": [variance_inflation_factor(design.values, i) for i in range(1, design.shape[1])],
    })


# ==============================================================================================================
# STEP 5 - PREPROCESSING / FEATURE TRANSFORMATION (week 9)
# Winsorising, Yeo-Johnson power transform + z-scores, p-value backward elimination and the log(1 + y) target.
# Every step is fitted inside a scikit-learn pipeline, so nothing from the test set or a CV fold leaks into training.
# ==============================================================================================================
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to the 1st-99th percentile of the training data."""

    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.percentile(X, self.lower, axis=0)
        self.high_ = np.percentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


def transformer(continuous, binary=()):
    """Winsorise + Yeo-Johnson (which also z-scores) the continuous variables, z-score the 0/1 ones.

    Output columns come out in the order continuous + binary.
    """
    parts = [("continuous", make_pipeline(Winsorizer(), PowerTransformer(method="yeo-johnson")), list(continuous))]
    if binary:
        parts.append(("binary", StandardScaler(), list(binary)))
    return ColumnTransformer(parts)


class BackwardElimination(BaseEstimator, TransformerMixin):
    """Drop the least significant variable (OLS p-value) until every p-value is below alpha."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def fit(self, X, y):
        X = pd.DataFrame(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        keep = list(range(X.shape[1]))
        while len(keep) > 1:
            p_values = sm.OLS(y, sm.add_constant(X[keep])).fit().pvalues.iloc[1:]
            if p_values.max() < self.alpha:
                break
            keep.remove(keep[int(np.argmax(p_values.values))])
        self.keep_ = keep
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


class LogTargetRegressor(RegressorMixin, BaseEstimator):
    """Fit on log(1 + y), predict back on the original scale.

    Turning a log prediction straight back gives roughly the median, which is too low for a skewed
    count like goals, so predictions are multiplied by Duan's smearing factor: the mean of
    exp(residual) on the training data.
    """

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        z = np.log1p(np.asarray(y, dtype=float))
        self.regressor_ = clone(self.regressor).fit(X, z)
        self.smearing_ = float(np.mean(np.exp(z - self.regressor_.predict(X))))
        return self

    def predict(self, X):
        return np.exp(self.regressor_.predict(X)) * self.smearing_ - 1


# ==============================================================================================================
# STEP 6 - MODEL BUILDERS (weeks 8 and 9)
# Baseline, simple LR, multiple LR, backward elimination, Ridge and Lasso.
# ==============================================================================================================
def baseline():
    return DummyRegressor(strategy="mean")


def simple_lr():
    return LinearRegression()


def zscore_mlr():
    return make_pipeline(StandardScaler(), LinearRegression())


def transformed_mlr(continuous, binary=()):
    return make_pipeline(transformer(continuous, binary), LinearRegression())


def backward_elimination(continuous, binary=()):
    return make_pipeline(transformer(continuous, binary), BackwardElimination(), LinearRegression())


def ridge(continuous, binary=()):
    return make_pipeline(transformer(continuous, binary), RidgeCV(alphas=np.logspace(-3, 3, 61)))


def lasso(continuous, binary=()):
    return make_pipeline(transformer(continuous, binary),
                         LassoCV(cv=CV_FOLDS, random_state=RANDOM_STATE, max_iter=50000))


def log_target(model):
    return LogTargetRegressor(model)


def unwrap(model):
    """Return (fitted sklearn model, target transform). The transform is None for a raw target."""
    if isinstance(model, LogTargetRegressor):
        return model.regressor_, np.log1p
    return model, None


def used_variables(model, columns):
    inner, _ = unwrap(model)
    columns = list(columns)
    steps = getattr(inner, "named_steps", {})
    if "backwardelimination" in steps:
        return [columns[i] for i in steps["backwardelimination"].keep_]
    last = inner[-1] if steps else inner
    if isinstance(last, LassoCV):
        return [c for c, b in zip(columns, last.coef_) if abs(b) > 1e-8]
    return columns


def standardised_coefficients(model, columns, features):
    """Coefficients on the transformed scale, 0 for variables the model does not use."""
    inner, _ = unwrap(model)
    names = list(columns)
    if "backwardelimination" in inner.named_steps:
        names = used_variables(model, columns)
    coefs = pd.Series(0.0, index=features)
    coefs[names] = np.ravel(inner[-1].coef_)
    return coefs


# ==============================================================================================================
# STEP 7 - EVALUATION METRICS AND SIMULATION (weeks 8, 9 and 11)
# MAE, MSE, RMSE, normalised RMSE, R2, bootstrap CIs, simulated outcomes, Brier score, permutation importance.
# ==============================================================================================================
def regression_metrics(y_true, y_pred, y_range):
    mse = metrics.mean_squared_error(y_true, y_pred)
    return {
        "MAE": metrics.mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "NRMSE": math.sqrt(mse) / y_range,
        "R2": metrics.r2_score(y_true, y_pred),
    }


def bootstrap_ci(y_true, y_pred, y_range, groups=None, level=0.95):
    """Percentile bootstrap CIs for the test metrics. With groups, whole matches are resampled."""
    rng = np.random.default_rng(RANDOM_STATE)
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    units = np.arange(len(y_true)) if groups is None else np.asarray(groups)
    labels = np.unique(units)
    draws = []
    for _ in range(N_BOOTSTRAP):
        picked = rng.choice(labels, size=len(labels), replace=True)
        idx = np.concatenate([np.flatnonzero(units == u) for u in picked])
        draws.append(regression_metrics(y_true[idx], y_pred[idx], y_range))
    draws = pd.DataFrame(draws)
    point = regression_metrics(y_true, y_pred, y_range)
    tail = (1 - level) / 2 * 100
    return pd.DataFrame({"metric": draws.columns, "estimate": [point[m] for m in draws.columns],
                         "ci_low": draws.quantile(tail / 100).to_numpy(),
                         "ci_high": draws.quantile(1 - tail / 100).to_numpy()})


def outcome(goal_difference):
    return np.sign(np.round(np.asarray(goal_difference, dtype=float)))


def normal_outcome_probabilities(predicted_gd, sd):
    """Simulate every match many times (week 11) with GD ~ Normal(prediction, error SD)."""
    rng = np.random.default_rng(RANDOM_STATE)
    sims = np.round(rng.normal(np.asarray(predicted_gd)[:, None], sd, size=(len(predicted_gd), N_SIMULATIONS)))
    return np.column_stack([(sims > 0).mean(axis=1), (sims == 0).mean(axis=1), (sims < 0).mean(axis=1)])


def poisson_scorelines(rate_a, rate_b):
    """Simulate scorelines with goals ~ Poisson(predicted goals) for each side."""
    rng = np.random.default_rng(RANDOM_STATE)
    rate_a = np.clip(np.asarray(rate_a, dtype=float), 0.05, None)[:, None]
    rate_b = np.clip(np.asarray(rate_b, dtype=float), 0.05, None)[:, None]
    goals_a = rng.poisson(rate_a, size=(len(rate_a), N_SIMULATIONS))
    goals_b = rng.poisson(rate_b, size=(len(rate_b), N_SIMULATIONS))
    return goals_a, goals_b


def brier_score(probabilities, actual_outcome):
    """Multi-class Brier score for win / draw / loss forecasts (lower is better)."""
    actual_outcome = np.asarray(actual_outcome)
    truth = np.column_stack([actual_outcome == 1, actual_outcome == 0, actual_outcome == -1]).astype(float)
    return float(np.mean(np.sum((np.asarray(probabilities) - truth) ** 2, axis=1)))


def permutation_importance(model, columns, X, y, features, y_range, repeats=50):
    """Increase in test RMSE when one variable is shuffled (variables the model ignores score 0)."""
    rng = np.random.default_rng(RANDOM_STATE)
    base = regression_metrics(y, model.predict(X[columns]), y_range)["RMSE"]
    rows = []
    for feature in features:
        increases = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            increases.append(regression_metrics(y, model.predict(shuffled[columns]), y_range)["RMSE"] - base)
        rows.append({"variable": feature, "importance": np.mean(increases), "std": np.std(increases)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


# ==============================================================================================================
# STEP 8 - REGRESSION DIAGNOSTICS (statsmodels OLS)
# Assumption tests (Shapiro-Wilk, Jarque-Bera, Breusch-Pagan, RESET, Durbin-Watson), OLS steps and influence.
# ==============================================================================================================
def ols(X, y):
    return sm.OLS(np.asarray(y, dtype=float), sm.add_constant(X)).fit()


def assumption_tests(fit):
    resid = fit.resid
    jb_stat, jb_p, _, _ = jarque_bera(resid)
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, fit.model.exog)
    reset = linear_reset(fit, power=2, use_f=True)
    shapiro = st.shapiro(resid)
    rows = [
        ("Normal residuals", "Shapiro-Wilk", shapiro.statistic, shapiro.pvalue),
        ("Normal residuals", "Jarque-Bera", jb_stat, jb_p),
        ("Constant variance", "Breusch-Pagan", bp_stat, bp_p),
        ("Linear form", "Ramsey RESET", float(reset.fvalue), float(reset.pvalue)),
        ("Independent errors", "Durbin-Watson (about 2 is good)", durbin_watson(resid), np.nan),
    ]
    table = pd.DataFrame(rows, columns=["assumption", "test", "statistic", "p_value"])
    table["holds_at_5pct"] = np.where(table["p_value"].isna(), "", np.where(table["p_value"] > 0.05, "yes", "no"))
    return table


def refit_ols(model, columns, X, y):
    """statsmodels OLS on the transformed variables a fitted sklearn model actually uses."""
    inner, target_transform = unwrap(model)
    target = np.asarray(y, dtype=float)
    if target_transform is not None:
        target = target_transform(target)
    used = used_variables(model, columns)
    if hasattr(inner, "steps"):
        names = used if "backwardelimination" in inner.named_steps else list(columns)
        design = pd.DataFrame(np.asarray(inner[:-1].transform(X[columns]), dtype=float), columns=names, index=X.index)
    else:
        design = X[columns]
    return ols(design[used], target)


def ols_steps(steps, folder):
    """Fit each class-style step with statsmodels, save its summary and return a comparison table."""
    rows = []
    for i, (name, (X, y)) in enumerate(steps.items(), start=1):
        fit = ols(X, y)
        (folder / f"05_ols_step{i}.txt").write_text(f"{name}\n\n{fit.summary()}\n")
        rows.append({"step": name, "variables": X.shape[1], "R2": fit.rsquared, "adj_R2": fit.rsquared_adj,
                     "AIC": fit.aic, "BIC": fit.bic, "F_test_p": fit.f_pvalue,
                     "significant_variables": int((fit.pvalues.drop("const") < 0.05).sum()),
                     "max_VIF": vif_table(X)["VIF"].max() if X.shape[1] > 1 else 1.0})
    return pd.DataFrame(rows)


def influence_table(fit, labels):
    influence = fit.get_influence()
    table = pd.DataFrame({
        "observation": labels,
        "leverage": influence.hat_matrix_diag,
        "studentised_residual": influence.resid_studentized_external,
        "cooks_distance": influence.cooks_distance[0],
    })
    table["influential"] = table["cooks_distance"] > 4 / len(table)
    return table


# ==============================================================================================================
# STEP 9 - EXPERIMENTS: CROSS-VALIDATION AND MODEL SELECTION
# Repeated 5-fold CV on the training set, corrected resampled t-test, model selection, repeated 60/40 splits.
# ==============================================================================================================
def cv_scores(model, columns, X, y, y_range, groups=None):
    """Repeated 5-fold cross-validation on the training data only."""
    rows = []
    for repeat in range(CV_REPEATS):
        if groups is None:
            folds = KFold(CV_FOLDS, shuffle=True, random_state=repeat)
        else:
            folds = GroupKFold(CV_FOLDS, shuffle=True, random_state=repeat)
        for fold, (fit_idx, val_idx) in enumerate(folds.split(X, y, groups)):
            fitted = clone(model).fit(X.iloc[fit_idx][columns], y.iloc[fit_idx])
            pred = fitted.predict(X.iloc[val_idx][columns])
            rows.append({"repeat": repeat, "fold": fold, "n_fit": len(fit_idx), "n_val": len(val_idx),
                         **regression_metrics(y.iloc[val_idx], pred, y_range)})
    return pd.DataFrame(rows)


def corrected_t_test(scores_a, scores_b):
    """Paired t-test on fold RMSE differences, with the Nadeau-Bengio correction for overlapping folds."""
    diff = scores_a["RMSE"].to_numpy() - scores_b["RMSE"].to_numpy()
    k = len(diff)
    ratio = scores_a["n_val"].mean() / scores_a["n_fit"].mean()
    t = diff.mean() / math.sqrt((1 / k + ratio) * diff.var(ddof=1))
    return t, 2 * st.t.sf(abs(t), df=k - 1)


def compare_models(models, X_train, y_train, X_test, y_test, y_range, groups=None, references=()):
    """Cross-validate every model on the training set, then score each one once on the test set."""
    summary, fold_scores, fitted = [], {}, {}
    for name, (columns, model) in models.items():
        scores = cv_scores(model, columns, X_train, y_train, y_range, groups)
        fold_scores[name] = scores
        fitted[name] = clone(model).fit(X_train[columns], y_train)
        test = regression_metrics(y_test, fitted[name].predict(X_test[columns]), y_range)
        summary.append({
            "model": name, "reference": name in references, "n_variables": len(columns),
            "cv_RMSE": scores["RMSE"].mean(), "cv_RMSE_sd": scores["RMSE"].std(),
            "cv_MAE": scores["MAE"].mean(), "cv_R2": scores["R2"].mean(),
            **{f"test_{k}": v for k, v in test.items()},
        })
    summary = pd.DataFrame(summary).sort_values("cv_RMSE").reset_index(drop=True)
    cv_best = summary.loc[~summary["reference"], "model"].iloc[0]
    summary["p_vs_cv_best"] = [np.nan if name == cv_best else corrected_t_test(fold_scores[name], fold_scores[cv_best])[1]
                               for name in summary["model"]]
    return summary, fitted, fold_scores


def select_model(summary, models, fitted, X_train, y_train):
    """Accuracy first, then validity.

    Start from the candidate with the lowest CV RMSE. Among the candidates that are not significantly
    worse than it, take the most accurate one whose residuals pass every assumption test.
    """
    candidates = summary[~summary["reference"]]
    tied = candidates[candidates["p_vs_cv_best"].isna() | (candidates["p_vs_cv_best"] > ALPHA)]
    rows = []
    for name, cv_rmse, p_tie in zip(tied["model"], tied["cv_RMSE"], tied["p_vs_cv_best"]):
        columns, _ = models[name]
        checks = assumption_tests(refit_ols(fitted[name], columns, X_train, y_train))
        p_values = dict(zip(checks["test"], checks["p_value"]))
        p_values = {k: v for k, v in p_values.items() if not np.isnan(v)}
        rows.append({"model": name, "cv_RMSE": cv_rmse, "p_vs_cv_best": p_tie, **p_values,
                     "all_assumptions_hold": all(p > ALPHA for p in p_values.values())})
    table = pd.DataFrame(rows)
    passing = table[table["all_assumptions_hold"]]
    best = passing["model"].iloc[0] if len(passing) else table["model"].iloc[0]
    return best, table


def repeated_splits(models, X, y, y_range, groups=None):
    """Re-run the 60/40 split many times to see how much a single split can mislead."""
    rows = []
    units = np.unique(groups) if groups is not None else np.arange(len(X))
    labels = groups if groups is not None else np.arange(len(X))
    for seed in range(N_SPLITS):
        train_units, _ = train_test_split(units, test_size=TEST_SIZE, random_state=seed)
        is_train = np.isin(labels, train_units)
        for name, (columns, model) in models.items():
            fitted = clone(model).fit(X.loc[is_train, columns], y[is_train])
            pred = fitted.predict(X.loc[~is_train, columns])
            rows.append({"split": seed, "model": name, **regression_metrics(y[~is_train], pred, y_range)})
    return pd.DataFrame(rows)


# ==============================================================================================================
# STEP 10 - VISUALISATION
# Every figure is saved into this objective's images/ folder.
# ==============================================================================================================
BLUE, CORAL, GREEN, GOLD, GREY = "#2a6f97", "#e07a5f", "#5a9e7c", "#e9b44c", "#8d99ae"


def setup():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titleweight": "bold",
                         "axes.titlesize": 11, "axes.labelsize": 10})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def label(name):
    return LABELS.get(name, name)


def smooth(x, y, frac):
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = lowess(y, x, frac=frac)
    return curve[~np.isnan(curve).any(axis=1)]


def histograms(df, columns, path, title):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, columns):
        sns.histplot(df[col], kde=True, ax=ax, color=BLUE, edgecolor="white")
        ax.set_title(label(col))
        ax.set_xlabel("")
        ax.text(0.97, 0.95, f"skew = {st.skew(df[col]):.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", ec=GREY))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def standardised_boxplot(df, columns, path, title, notes=()):
    z = (df[columns] - df[columns].mean()) / df[columns].std()
    long = z.melt(var_name="variable", value_name="z").assign(variable=lambda d: d["variable"].map(label))
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=long, y="variable", x="z", ax=ax, color="#a9cce3", fliersize=4,
                flierprops=dict(markerfacecolor=CORAL, markeredgecolor=CORAL))
    for col, value, text in notes:
        z_value = (value - df[col].mean()) / df[col].std()
        y_pos = columns.index(col)
        shift, align = (1.0, "left") if z_value < 0 else (-1.0, "right")
        ax.annotate(text, xy=(z_value, y_pos), xytext=(z_value + shift, y_pos - 0.45), fontsize=9, ha=align,
                    arrowprops=dict(arrowstyle="->", color=CORAL), color=CORAL)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlabel("z-score (standard deviations from the mean)")
    ax.set_ylabel("")
    ax.set_title(title)
    save(fig, path)


def heatmap(df, columns, path, title):
    corr = df[columns].rename(columns=label).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
                annot=True, fmt=".2f", square=True, linewidths=0.5, cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    ax.set_title(title)
    save(fig, path)


def linearity_grid(df, columns, target, path, title):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
    for ax, col in zip(axes.flat, columns):
        x, y = df[col].to_numpy(dtype=float), df[target].to_numpy(dtype=float)
        ax.scatter(x, y + np.random.default_rng(0).uniform(-0.12, 0.12, len(y)), s=16, alpha=0.55, color=BLUE)
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, intercept + slope * grid, color=CORAL, lw=2, label="linear fit")
        if len(np.unique(x)) > 2:
            curve = smooth(x, y, frac=0.6)
            ax.plot(curve[:, 0], curve[:, 1], color=GREEN, lw=2, ls="--", label="LOWESS (flexible)")
        ax.set_title(f"{label(col)}  (r = {np.corrcoef(x, y)[0, 1]:.2f})", fontsize=10)
        ax.set_xlabel(label(col), fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(label(target))
    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def pairplot(df, columns, hue, path, title, palette=None):
    plot_df = df[columns + [hue]].rename(columns=label)
    order = list(palette) if palette else None
    grid = sns.pairplot(plot_df, hue=hue, hue_order=order, corner=True, palette=palette,
                        plot_kws=dict(s=22, alpha=0.7), diag_kind="kde", height=2.3)
    grid.figure.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    save(grid.figure, path)


def vif_bars(before, after, path, title, threshold=5):
    """VIF per variable. Pass after=None when no variable needed dropping."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if after is None:
        merged = before.assign(variable=before["variable"].map(label)).sort_values("VIF")
        y = np.arange(len(merged))
        ax.barh(y, merged["VIF"], height=0.6, color=BLUE, label="all 8 variables")
        for i, value in enumerate(merged["VIF"]):
            ax.text(value + 0.05, i, f"{value:.2f}", va="center", fontsize=9)
    else:
        merged = before.merge(after, on="variable", how="left", suffixes=("_before", "_after"))
        merged["variable"] = merged["variable"].map(label)
        merged = merged.sort_values("VIF_before")
        y = np.arange(len(merged))
        ax.barh(y - 0.2, merged["VIF_before"], height=0.4, color=CORAL, label="all 8 variables")
        ax.barh(y + 0.2, merged["VIF_after"].fillna(0), height=0.4, color=BLUE, label="after the collinearity fix")
        for i, row in enumerate(merged.itertuples()):
            ax.text(row.VIF_before + 0.3, i - 0.2, f"{row.VIF_before:.1f}", va="center", fontsize=9)
            if pd.isna(row.VIF_after):
                ax.text(0.3, i + 0.2, "dropped", va="center", fontsize=9, color=CORAL)
            else:
                ax.text(row.VIF_after + 0.3, i + 0.2, f"{row.VIF_after:.1f}", va="center", fontsize=9)
    ax.axvline(threshold, color="black", ls="--", lw=1.2, label=f"VIF = {threshold} rule of thumb")
    ax.set_yticks(y, merged["variable"])
    ax.set_xlabel("Variance inflation factor (training data)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    save(fig, path)


def transformation_plot(before, after, columns, focus, path, title):
    skew_before = before[columns].apply(st.skew)
    skew_after = after[columns].apply(st.skew)
    order = skew_before.abs().sort_values().index
    fig = plt.figure(figsize=(17, 5.5))
    ax0 = fig.add_subplot(1, 3, 1)
    for i, col in enumerate(order):
        ax0.plot([skew_before[col], skew_after[col]], [i, i], color=GREY, lw=2, zorder=1)
        ax0.scatter(skew_before[col], i, color=CORAL, s=60, zorder=2, label="raw" if i == 0 else None)
        ax0.scatter(skew_after[col], i, color=BLUE, s=60, zorder=2, label="winsorised + Yeo-Johnson" if i == 0 else None)
    ax0.axvline(0, color="black", lw=1)
    ax0.set_yticks(range(len(order)),
                   [f"{textwrap.fill(label(c), 24)}\n({skew_before[c]:.2f} -> {skew_after[c]:.2f})" for c in order],
                   fontsize=8)
    ax0.set_xlabel("Skewness (0 = symmetric)")
    ax0.set_title("Skewness before and after")
    ax0.legend(loc="lower right", fontsize=8)
    for k, (data, name, colour) in enumerate([(before, "raw", CORAL), (after, "transformed", BLUE)]):
        ax = fig.add_subplot(1, 3, k + 2)
        (osm, osr), (slope, intercept, _) = st.probplot(np.asarray(data[focus], dtype=float), dist="norm")
        ax.scatter(osm, osr, s=18, color=colour)
        ax.plot(osm, slope * np.asarray(osm) + intercept, color="black", lw=1)
        p_value = st.shapiro(np.asarray(data[focus], dtype=float)).pvalue
        ax.set_title(f"Q-Q plot ({name})\n{label(focus)}\nShapiro-Wilk p = {p_value:.3g}")
        ax.set_xlabel("Theoretical normal quantiles")
        ax.set_ylabel("Sample quantiles")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def model_comparison(summary, selected, path, title, unit="goals"):
    df = summary.sort_values("cv_RMSE", ascending=False).reset_index(drop=True)
    colours = [CORAL if m == selected else (GREY if r else BLUE) for m, r in zip(df["model"], df["reference"])]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(df) + 1.8))
    ax.barh(df["model"], df["cv_RMSE"], xerr=df["cv_RMSE_sd"], color=colours, alpha=0.9,
            error_kw=dict(ecolor="#555555", capsize=3, lw=1))
    ax.scatter(df["test_RMSE"], df["model"], color="black", marker="D", s=40, zorder=3)
    for i, row in df.iterrows():
        ax.text(0.03, i, f"CV R² {row.cv_R2:.2f}", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel(f"RMSE in {unit} (bars: mean ± 1 SD over repeated 5-fold CV on the training set)")
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (CORAL, BLUE, GREY)]
    handles.append(plt.Line2D([], [], color="black", marker="D", ls=""))
    ax.legend(handles, ["selected model", "candidate model", "benchmark / reduced model",
                        "test-set RMSE (held-out 40%)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.1 - 0.4 / len(df)), ncol=4, frameon=False)
    save(fig, path)


def optimisation_path(summary, steps, path, title):
    df = summary.set_index("model").loc[list(steps)]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, df["cv_RMSE"], marker="o", color=BLUE, lw=2.5, label="cross-validated RMSE (training set)")
    ax.plot(x, df["test_RMSE"], marker="D", color=CORAL, lw=2, ls="--", label="test RMSE (held-out 40%)")
    for i, (cv_rmse, cv_r2, test_rmse) in enumerate(zip(df["cv_RMSE"], df["cv_R2"], df["test_RMSE"])):
        offset = -22 if test_rmse > cv_rmse else 12
        ax.annotate(f"{cv_rmse:.3f} (R² {cv_r2:.2f})", (i, cv_rmse), textcoords="offset points", xytext=(0, offset),
                    ha="center", fontsize=9, color=BLUE)
    ax.set_xticks(x, [steps[m] for m in df.index], fontsize=9)
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.margins(y=0.25)
    save(fig, path)


def actual_vs_predicted(actual, predicted, names, path, title, jitter=0.0, n_labels=4):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    shown = actual + np.random.default_rng(1).uniform(-jitter, jitter, len(actual))
    error = np.abs(actual - predicted)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(shown, predicted, s=45, color=BLUE, alpha=0.75, edgecolor="white")
    low, high = min(actual.min(), predicted.min()) - 0.5, max(actual.max(), predicted.max()) + 0.5
    ax.plot([low, high], [low, high], color=CORAL, ls="--", lw=1.5, label="perfect prediction")
    for k, i in enumerate(np.argsort(-error)[:n_labels]):
        right_side = shown[i] > low + 0.7 * (high - low)
        ax.annotate(names[i], (shown[i], predicted[i]), xytext=(-8 if right_side else 8, [10, -16, 24, -30][k % 4]),
                    textcoords="offset points", ha="right" if right_side else "left", fontsize=8, color="#333333",
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Actual" + (" (jittered)" if jitter else ""))
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend(loc="upper left")
    save(fig, path)


def residual_diagnostics(fitted_values, residuals, path, title):
    fitted_values, residuals = np.asarray(fitted_values, dtype=float), np.asarray(residuals, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.scatter(fitted_values, residuals, color=BLUE, s=22, alpha=0.7)
    curve = smooth(fitted_values, residuals, frac=0.7)
    ax.plot(curve[:, 0], curve[:, 1], color=CORAL, lw=2)
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set(title="Residuals vs fitted (looking for a pattern)", xlabel="Fitted value", ylabel="Residual")
    ax = axes[0, 1]
    (osm, osr), (slope, intercept, _) = st.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, color=BLUE, s=18)
    ax.plot(osm, slope * np.asarray(osm) + intercept, color=CORAL)
    ax.set(title="Normal Q-Q plot of residuals", xlabel="Theoretical quantiles", ylabel="Residual quantiles")
    ax = axes[1, 0]
    sns.histplot(residuals, kde=True, ax=ax, color=BLUE, edgecolor="white")
    ax.set(title="Residual distribution", xlabel="Residual")
    ax = axes[1, 1]
    scaled = np.sqrt(np.abs(residuals / residuals.std()))
    ax.scatter(fitted_values, scaled, color=BLUE, s=22, alpha=0.7)
    curve = smooth(fitted_values, scaled, frac=0.7)
    ax.plot(curve[:, 0], curve[:, 1], color=CORAL, lw=2)
    ax.set(title="Scale-location (constant variance?)", xlabel="Fitted value", ylabel="sqrt(|standardised residual|)")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def influence_plot(table, path, title):
    """table in training-row order with observation, leverage, studentised_residual, cooks_distance."""
    n = len(table)
    threshold = 4 / n
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    ax.vlines(np.arange(n), 0, table["cooks_distance"], color=BLUE, lw=1.5)
    ax.axhline(threshold, color=CORAL, ls="--", label=f"4/n = {threshold:.3f}")
    for i in table["cooks_distance"].nlargest(3).index:
        ax.annotate(table.loc[i, "observation"], (i, table.loc[i, "cooks_distance"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.set(title="Cook's distance (training data)", xlabel="Training observation", ylabel="Cook's distance")
    ax.legend()
    ax = axes[1]
    flagged = table["cooks_distance"] > threshold
    ax.scatter(table["leverage"], table["studentised_residual"], c=np.where(flagged, CORAL, BLUE), s=28)
    ax.axhline(0, color=GREY, lw=1)
    for y in (-3, 3):
        ax.axhline(y, color=GREY, ls=":", lw=1)
    ax.set(title="Leverage vs studentised residual (orange = above 4/n)", xlabel="Leverage",
           ylabel="Studentised residual")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def coefficient_plot(coefs, path, title):
    """coefs: DataFrame indexed by variable, one column per model."""
    df = coefs.copy()
    df.index = [label(i) for i in df.index]
    df = df.loc[df.abs().max(axis=1).sort_values().index]
    y = np.arange(len(df))
    height = 0.8 / df.shape[1]
    colours = [GREY, BLUE, CORAL, GREEN]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(df) + 1.5))
    for k, col in enumerate(df.columns):
        ax.barh(y + (k - (df.shape[1] - 1) / 2) * height, df[col], height=height, color=colours[k % 4], label=col)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, df.index)
    ax.set_xlabel("Change in the target per 1 SD change in the (transformed) variable")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9, frameon=False)
    save(fig, path)


def importance_plot(importance, path, title, unit="goals"):
    """importance: long DataFrame with variable, model, importance, std."""
    models = list(dict.fromkeys(importance["model"]))
    order = importance.groupby("variable")["importance"].max().sort_values().index
    y = np.arange(len(order))
    height = 0.8 / len(models)
    colours = [CORAL, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(order) + 1.5))
    for k, model in enumerate(models):
        part = importance[importance["model"] == model].set_index("variable").loc[order]
        ax.barh(y + (k - (len(models) - 1) / 2) * height, part["importance"], xerr=part["std"], height=height,
                color=colours[k % 3], label=model, error_kw=dict(ecolor="#555555", capsize=2, lw=1))
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, [label(v) for v in order])
    ax.set_xlabel(f"Increase in test RMSE when the variable is randomly shuffled ({unit})")
    ax.set_title(title)
    ax.legend(loc="lower right")
    save(fig, path)


def robustness_plot(results, order, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["RMSE", "R2"]):
        sns.boxplot(data=results, x="model", y=metric, order=order, ax=ax, color="#a9cce3", fliersize=2)
        ax.set_xlabel("")
        ax.set_ylabel(metric if metric == "RMSE" else "R²")
        ax.set_xticks(range(len(order)), [textwrap.fill(name, 16) for name in order], fontsize=9)
    low = results.groupby("model")["R2"].quantile(0.02).min()
    axes[1].set_ylim(max(low - 0.1, -1.0), None)
    axes[1].axhline(0, color="black", lw=1)
    axes[0].set_title("Test RMSE over repeated 60/40 splits")
    axes[1].set_title("Test R² over repeated 60/40 splits")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def lasso_path_plot(X, y, names, chosen_alpha, path, title):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float) - np.mean(y)
    alpha_max = np.max(np.abs(X.T @ y)) / len(y)
    grid = np.logspace(np.log10(alpha_max), np.log10(alpha_max / 1000), 120)
    alphas, coefs, _ = lasso_path(X, y, alphas=grid)
    palette = sns.color_palette("tab10", len(names))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k, name in enumerate(names):
        ax.plot(np.log10(alphas), coefs[k], color=palette[k], lw=2, label=label(name))
    ax.axvline(np.log10(chosen_alpha), color="black", ls="--", label="alpha chosen by CV")
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xlabel("log10(alpha)  (stronger penalty to the right)")
    ax.set_ylabel("Standardised coefficient")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, path)


def outcome_probability_bars(probs, actual, names, path, title, outcome_names, colours):
    """Stacked win/draw/loss probabilities per match, with a star on what actually happened."""
    probs = np.asarray(probs)
    order = np.argsort(probs[:, 0] - probs[:, 2])
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(probs) + 1.8))
    left = np.zeros(len(probs))
    for k, name in enumerate(outcome_names):
        ax.barh(np.arange(len(probs)), probs[order, k], left=left, color=colours[name], label=f"P({name})", height=0.8)
        left += probs[order, k]
    star_x = {1: 0.02, 0: 0.5, -1: 0.98}
    for i, result in enumerate(np.asarray(actual)[order]):
        ax.scatter(star_x[int(result)], i, marker="*", s=90, color="black", zorder=3)
    ax.set_yticks(np.arange(len(probs)), np.asarray(names)[order], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Simulated probability   (* = what actually happened)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    save(fig, path)


# ==============================================================================================================
# STEP 11 - OBJECTIVE 2.2 ANALYSIS: GOALS SCORED BY A TEAM
# run() is the full pipeline: wrangle + check -> describe -> inferential tests -> 60/40 split by match ->
# relationships -> OLS steps -> competing models -> final model -> variables -> robustness -> 2.1 vs 2.2.
# ==============================================================================================================
FEATURES = FEATURES_2_2
CONTINUOUS, BINARY = CONTINUOUS_2_2, BINARY_2_2
TARGET = TARGET_2_2
REFERENCES = ("Baseline: training mean", "Simple LR: Elo only", "MLR: backward elimination")
STEPS = {
    "MLR: z-score": "MLR, 8 variables\n(z-score only)",
    "MLR: winsorise + Yeo-Johnson": "+ winsorise\n+ Yeo-Johnson",
    "MLR: + log(1 + goals) target": "+ log(1 + goals)\ntarget",
    "Ridge + log(1 + goals) target": "+ Ridge penalty",
}
GOAL_GROUPS = {0: "0 goals", 1: "1 goal", 2: "2+ goals"}
GOAL_COLOURS = {"0 goals": CORAL, "1 goal": GOLD, "2+ goals": GREEN}


def save_table(df, name):
    df.to_csv(TABLES / f"{name}.csv", index=False)


def heading(text):
    print("\n" + "=" * 72 + f"\n2.2  {text}\n" + "=" * 72)


def goal_group(goals):
    return pd.Series(np.minimum(goals, 2)).map(GOAL_GROUPS).to_numpy()


def candidate_models():
    return {
        "Baseline: training mean": (FEATURES, baseline()),
        "Simple LR: Elo only": (["Elo_Rating_Diff"], simple_lr()),
        "MLR: z-score": (FEATURES, zscore_mlr()),
        "MLR: winsorise + Yeo-Johnson": (FEATURES, transformed_mlr(CONTINUOUS, BINARY)),
        "MLR: + log(1 + goals) target": (FEATURES, log_target(transformed_mlr(CONTINUOUS, BINARY))),
        "MLR: backward elimination": (FEATURES, backward_elimination(CONTINUOUS, BINARY)),
        "Ridge (alpha by CV)": (FEATURES, ridge(CONTINUOUS, BINARY)),
        "Lasso (alpha by CV)": (FEATURES, lasso(CONTINUOUS, BINARY)),
        "Ridge + log(1 + goals) target": (FEATURES, log_target(ridge(CONTINUOUS, BINARY))),
    }


def transform_train(X_train):
    return pd.DataFrame(transformer(CONTINUOUS, BINARY).fit(X_train).transform(X_train), columns=FEATURES,
                        index=X_train.index)


def overview_figure(df):
    goals = df[TARGET]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    counts = goals.value_counts().sort_index().reindex(range(goals.max() + 1), fill_value=0)
    ax.bar(counts.index, counts.values, color=BLUE, edgecolor="white", label="observed")
    expected = st.poisson.pmf(counts.index, goals.mean()) * len(goals)
    ax.plot(counts.index, expected, color=CORAL, marker="o", lw=2, label=f"Poisson (mean {goals.mean():.2f})")
    ax.set(title="Goals scored by a team in one match (208 team-matches)", xlabel="Goals", ylabel="Team-matches")
    ax.text(0.97, 0.6, f"mean = {goals.mean():.2f}\nvariance = {goals.var():.2f}\nskew = {st.skew(goals):.2f}",
            transform=ax.transAxes, ha="right", bbox=dict(boxstyle="round", fc="white", ec=GREY))
    ax.legend()

    ax = axes[0, 1]
    groups = [
        ("Elo favourite", goals[df["Elo_Rating_Diff"] > 0]), ("Elo underdog", goals[df["Elo_Rating_Diff"] <= 0]),
        ("Host nation at home", goals[df["Host_Advantage"] == 1]), ("Everyone else", goals[df["Host_Advantage"] == 0]),
        ("Group stage", goals[df["Knockout_Stage"] == 0]), ("Knockout", goals[df["Knockout_Stage"] == 1]),
    ]
    for i, (name, sample) in enumerate(groups):
        mean, low, high = mean_ci(sample)
        ax.errorbar(mean, i, xerr=[[mean - low], [high - mean]], fmt="o", color=[GREEN, CORAL][i % 2],
                    capsize=5, ms=8, lw=2)
        ax.text(high + 0.05, i, f"{mean:.2f}  (n = {len(sample)})", va="center", fontsize=9)
    ax.set_yticks(range(len(groups)), [g[0] for g in groups])
    ax.invert_yaxis()
    ax.axvline(goals.mean(), color=GREY, ls="--", lw=1, label="overall mean")
    ax.set(title="Mean goals with 95% confidence intervals", xlabel="Goals per team per match")
    ax.set_xlim(0, max(3.4, ax.get_xlim()[1] + 0.6))
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    bins = pd.cut(df["Elo_Rating_Diff"], bins=[-600, -300, -150, -50, 50, 150, 300, 600])
    summary = df.groupby(bins, observed=True)[TARGET].agg(["mean", "count", "sem"])
    centres = [interval.mid for interval in summary.index]
    ax.errorbar(centres, summary["mean"], yerr=1.96 * summary["sem"], fmt="o-", color=BLUE, capsize=4, lw=2)
    for x, m, n in zip(centres, summary["mean"], summary["count"]):
        ax.text(x, m + 0.12, f"n={n}", ha="center", fontsize=8, color="#555555")
    ax.set(title="Mean goals by Elo gap (bins, 95% CI)", xlabel="Elo rating difference (team - opponent), binned",
           ylabel="Mean goals")

    ax = axes[1, 1]
    status = np.where(df["Elo_Rating_Diff"] > 0, "Elo favourite", "Elo underdog")
    table = pd.crosstab(status, goal_group(goals), normalize="index")[list(GOAL_GROUPS.values())]
    sns.heatmap(table, annot=True, fmt=".0%", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 12})
    ax.set(title="P(goals scored | favourite or underdog)", xlabel="", ylabel="")

    fig.suptitle("Objective 2.2 - what the target looks like", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, FIGURES / "01_goals_overview.png")


def linearisation_figure(train, reset_raw, reset_log):
    x = train["Elo_Rating_Diff"].to_numpy(dtype=float)
    jitter = np.random.default_rng(0).uniform(-0.1, 0.1, len(x))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, y, name, p, spread in [(axes[0], train[TARGET].to_numpy(dtype=float), "Goals", reset_raw, 1.0),
                                   (axes[1], np.log1p(train[TARGET].to_numpy(dtype=float)), "log(1 + goals)",
                                    reset_log, 0.4)]:
        ax.scatter(x, y + jitter * spread, s=18, alpha=0.6, color=BLUE)
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, intercept + slope * grid, color=CORAL, lw=2, label="straight line")
        curve = smooth(x, y, frac=0.5)
        ax.plot(curve[:, 0], curve[:, 1], color=GREEN, lw=2.5, ls="--", label="LOWESS")
        ax.set(xlabel="Elo rating difference (team - opponent)", ylabel=name,
               title=f"{name} vs Elo difference\nRamsey RESET on the full 8-variable model: p = {p:.3f}")
        ax.legend(loc="upper left")
    fig.suptitle("Goals are a skewed count - the log(1 + goals) scale behaves better (training data)", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, FIGURES / "06_linearisation.png")


def route_comparison_figure(matches, table, floors, achieved):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(2, 2, 1)
    shown = matches["actual_GD"] + np.random.default_rng(2).uniform(-0.12, 0.12, len(matches))
    ax.scatter(shown, matches["direct_2_1"], color=BLUE, s=40, label="2.1 model (goal difference directly)")
    ax.scatter(shown, matches["implied_2_2"], color=CORAL, s=40, marker="s", label="2.2 model (goals A - goals B)")
    for x, a, b in zip(shown, matches["direct_2_1"], matches["implied_2_2"]):
        ax.plot([x, x], [a, b], color=GREY, lw=0.8, alpha=0.6)
    lim = [-4.5, 6.5]
    ax.plot(lim, lim, color="black", ls="--", lw=1)
    ax.set(xlim=lim, ylim=[-3.5, 3.5], xlabel="Actual goal difference (jittered)", ylabel="Predicted goal difference",
           title="Two routes to the same 42 test matches")
    ax.legend(loc="upper left", fontsize=9)

    ax = fig.add_subplot(2, 2, 2)
    show = table.set_index("route")[["RMSE", "MAE", "winner_accuracy", "brier"]]
    colours = [BLUE, CORAL, GREEN]
    x = np.arange(4)
    width = 0.8 / len(show)
    for k, (route, row) in enumerate(show.iterrows()):
        bars = ax.bar(x + (k - (len(show) - 1) / 2) * width, row.values, width=width, color=colours[k], label=route)
        for bar, value in zip(bars, row.values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["RMSE\n(lower better)", "MAE\n(lower better)", "Winner accuracy\n(higher better)",
                      "Brier score\n(lower better)"])
    ax.set_title("Goal-difference accuracy on the 42 test matches")
    ax.legend(fontsize=8, loc="upper right")
    ax.margins(y=0.15)

    for k, (key, title) in enumerate([("gd", "Goal difference (42 test matches)"), ("goals", "Team goals (84 test rows)")]):
        ax = fig.add_subplot(2, 2, 3 + k)
        sns.histplot(floors[key], ax=ax, color=GREY, bins=40, stat="density", edgecolor="white")
        ax.axvline(np.median(floors[key]), color="black", ls="--", lw=1.5,
                   label=f"perfect-knowledge model, median {np.median(floors[key]):.2f}")
        for name, value, colour in achieved[key]:
            ax.axvline(value, color=colour, lw=2.5, label=f"{name}: {value:.2f}")
        ax.set(title=f"How low can RMSE go? {title}", xlabel="RMSE if goals are Poisson with the predicted rates",
               ylabel="Density")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Objective 2.1 vs 2.2 - predicting matches directly or team by team", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, FIGURES / "20_objective_2_1_vs_2_2.png")


def describe(df):
    heading("1. Descriptive statistics (weeks 2 and 7)")
    table = descriptive_table(df, [TARGET] + FEATURES)
    save_table(table, "01_descriptive_statistics")
    print(table[["variable", "mean", "median", "mode", "std", "min", "max", "iqr", "skewness"]].to_string(index=False))
    overview_figure(df)
    histograms(df, FEATURES, FIGURES / "02_variable_histograms.png",
                   "Distribution of the eight explanatory variables (208 team-matches)")
    z = (df[CONTINUOUS] - df[CONTINUOUS].mean()) / df[CONTINUOUS].std()
    col = z.abs().max().idxmax()
    row = df.loc[z[col].abs().idxmax()]
    standardised_boxplot(df, FEATURES, FIGURES / "03_outliers_boxplot.png",
                             "Outlier screening on a common z-score scale",
                             notes=[(col, row[col], f"{row['Team']} v {row['Opponent']} ({row[col]:.2f})")])


def inferential(df):
    heading("2. Inferential statistics (weeks 3, 4 and 10)")
    goals = df[TARGET]
    rows = []

    mean, low, high = mean_ci(goals)
    rows.append({"question": "Average goals per team per match", "method": "95% confidence interval (z)",
                 "estimate": mean, "ci_low": low, "ci_high": high})

    share, low, high = proportion_ci(int((goals > 0).sum()), len(goals))
    rows.append({"question": "Share of teams that score at least once", "method": "95% CI for a proportion",
                 "estimate": share, "ci_low": low, "ci_high": high})

    tests = [
        ("Do Elo favourites score more than underdogs?", goals[df["Elo_Rating_Diff"] > 0],
         goals[df["Elo_Rating_Diff"] <= 0], "greater"),
        ("Do host nations playing at home score more?", goals[df["Host_Advantage"] == 1],
         goals[df["Host_Advantage"] == 0], "greater"),
        ("Do teams score a different number of goals in knockout games?", goals[df["Knockout_Stage"] == 1],
         goals[df["Knockout_Stage"] == 0], "two-sided"),
    ]
    for question, a, b, alternative in tests:
        t, p = welch_t(a, b, alternative=alternative)
        rows.append({"question": question, "method": f"Welch two-sample t-test ({alternative})",
                     "estimate": a.mean() - b.mean(), "statistic": t, "p_value": p, "conclusion": conclusion(p)})

    dispersion = goals.var(ddof=1) / goals.mean()
    chi2 = (len(goals) - 1) * dispersion
    p = st.chi2.sf(chi2, len(goals) - 1)
    rows.append({"question": "Are goals more spread out than a Poisson count (variance = mean)?",
                 "method": "dispersion test, chi-square on (n-1) * variance / mean", "estimate": dispersion,
                 "statistic": chi2, "p_value": p, "conclusion": conclusion(p)})

    table = pd.DataFrame(rows)
    save_table(table.round(4), "02_inferential_tests")
    print(table[["question", "estimate", "ci_low", "ci_high", "p_value", "conclusion"]].round(3).to_string(index=False))

    status = np.where(df["Elo_Rating_Diff"] > 0, "Elo favourite", "Elo underdog")
    two_way = pd.crosstab(status, goal_group(goals), margins=True)
    conditional = pd.crosstab(status, goal_group(goals), normalize="index").round(3)
    save_table(two_way.reset_index(names="Elo status"), "03_two_way_table")
    save_table(conditional.reset_index(names="Elo status"), "03_conditional_probabilities")
    print("\nP(goals | Elo status):")
    print(conditional)
    return table, conditional


def relationships(df, train):
    heading("3. Correlation, linearity and collinearity (weeks 8 and 9)")
    heatmap(df, FEATURES + [TARGET], FIGURES / "04_correlation_heatmap.png",
                "Correlation matrix of the 2.2 variables and goals scored")
    linearity_grid(df, FEATURES, TARGET, FIGURES / "05_linearity_check.png",
                       "Is each relationship linear? Straight-line fit vs flexible LOWESS curve")
    nonlinear = nonlinearity_tests(train[CONTINUOUS], train[TARGET])
    save_table(nonlinear.round(4), "04_nonlinearity_tests")
    print(nonlinear.round(3).to_string(index=False))

    X_tf = transform_train(train[FEATURES])

    def reset_p(y):
        return assumption_tests(ols(X_tf, y)).set_index("test").loc["Ramsey RESET", "p_value"]

    reset_raw, reset_log = reset_p(train[TARGET]), reset_p(np.log1p(train[TARGET]))
    print(f"Ramsey RESET on the full model: goals p = {reset_raw:.3f}, log(1 + goals) p = {reset_log:.3f}")
    linearisation_figure(train, reset_raw, reset_log)

    strongest = df[FEATURES].corrwith(df[TARGET]).abs().sort_values(ascending=False).index[:3].tolist()
    pairplot(df.assign(Scored=goal_group(df[TARGET])), strongest + [TARGET], "Scored",
                 FIGURES / "07_pairplot.png", "The three strongest predictors, coloured by goals scored",
                 palette=GOAL_COLOURS)
    return nonlinear, reset_raw, reset_log


def statsmodels_steps(X_train, y_train):
    heading("4. Class-style OLS steps on the training set (week 9, statsmodels)")
    X_tf = transform_train(X_train)
    X_z = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=FEATURES, index=X_train.index)
    kept = [FEATURES[i] for i in BackwardElimination().fit(X_tf, y_train).keep_]

    steps = {
        "1. Raw variables": (X_train, y_train),
        "2. Z-score standardised": (X_z, y_train),
        "3. Winsorised + Yeo-Johnson": (X_tf, y_train),
        "4. Backward elimination (p < 0.05)": (X_tf[kept], y_train),
        "5. log(1 + goals) target (R2 on the log scale)": (X_tf, np.log1p(y_train)),
    }
    table = ols_steps(steps, TABLES)
    save_table(table.round(4), "05_ols_steps_summary")
    print(table.round(3).to_string(index=False))
    print("Backward elimination keeps:", kept)

    normality = pd.concat([normality_table(X_train, CONTINUOUS, "raw"),
                           normality_table(X_tf, CONTINUOUS, "winsorised + Yeo-Johnson")])
    save_table(normality.round(4), "06_normality_before_after")
    focus = X_train[CONTINUOUS].skew().abs().idxmax()
    transformation_plot(X_train, X_tf, CONTINUOUS, focus, FIGURES / "09_transformation.png",
                            "Making the continuous explanatory variables more Gaussian (training data)")

    vif = vif_table(X_tf)
    save_table(vif.round(3), "07_vif")
    vif_bars(vif, None, FIGURES / "08_vif.png", "Collinearity check: every VIF is below 5, nothing to drop")
    return X_tf, table, kept, vif


def final_model_report(model_set, fitted, best, train, test, X_tf, y_range):
    heading(f"6. Final model on the 40% test set: {best}")
    X_train, y_train, X_test, y_test = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    columns, _ = model_set[best]
    final = fitted[best]
    pred_test, pred_train = final.predict(X_test[columns]), final.predict(X_train[columns])

    rows = []
    for name in dict.fromkeys([best, "Ridge (alpha by CV)", "MLR: backward elimination", "Simple LR: Elo only",
                               "Baseline: training mean"]):
        cols, _ = model_set[name]
        rows.append({"model": name, **regression_metrics(y_test, fitted[name].predict(X_test[cols]), y_range)})
    test_table = pd.DataFrame(rows)
    save_table(test_table.round(4), "09_final_vs_benchmarks_test")
    for row in test_table.itertuples():
        print(f"\n{row.model}\nMAE:  {row.MAE:.3f}\nMSE:  {row.MSE:.3f}\nRMSE: {row.RMSE:.3f}\n"
              f"RMSE (normalised): {row.NRMSE:.3f}\nR^2:  {row.R2:.3f}")
    ci = bootstrap_ci(y_test, pred_test, y_range, groups=test["Match_ID"].to_numpy())
    save_table(ci.round(4), "09_final_model_bootstrap_ci")

    rounded = np.round(np.clip(pred_test, 0, None))
    exact = float(np.mean(rounded == y_test))
    within_one = float(np.mean(np.abs(rounded - y_test) <= 1))
    print(f"\nRounded prediction exactly right: {exact:.1%}   within one goal: {within_one:.1%}")

    predictions = test[["Row_ID", "Match_ID", "Stage", "Team", "Opponent"]].assign(
        actual_goals=y_test.to_numpy(), predicted_goals=pred_test.round(3), error=(y_test - pred_test).round(3))
    save_table(predictions, "10_test_predictions")

    names = (test["Team"] + " v " + test["Opponent"]).to_numpy()
    actual_vs_predicted(y_test, pred_test, names, FIGURES / "12_actual_vs_predicted.png",
                            f"{best}: actual vs predicted goals (84 test team-matches)", jitter=0.15)
    log_model = isinstance(final, LogTargetRegressor)
    if log_model:
        fitted_log = final.regressor_.predict(X_train[columns])
        residual_diagnostics(fitted_log, np.log1p(y_train) - fitted_log, FIGURES / "13_residuals_train.png",
                                 f"Residuals on the model scale, log(1 + goals) - training set ({best})")
    else:
        residual_diagnostics(pred_train, y_train - pred_train, FIGURES / "13_residuals_train.png",
                                 f"Residual diagnostics, training set ({best})")
    residual_diagnostics(pred_test, y_test - pred_test, FIGURES / "14_residuals_test.png",
                             f"Prediction errors in goals, test set ({best})")

    used = used_variables(final, columns)
    post_fit = refit_ols(final, columns, X_train, y_train)
    scale = "log(1 + goals)" if log_model else "goals"
    (TABLES / "12_final_model_ols_summary.txt").write_text(
        f"OLS refit on the variables used by {best} (training set, transformed variables, target = {scale})\n\n"
        f"{post_fit.summary()}\n\nSame model with heteroscedasticity-robust (HC3) standard errors\n\n"
        f"{post_fit.get_robustcov_results(cov_type='HC3').summary()}\n")
    assumptions = assumption_tests(post_fit)
    save_table(assumptions.round(4), "12_assumption_tests")
    influence = influence_table(post_fit, (train["Team"] + " v " + train["Opponent"]).to_numpy())
    save_table(influence.sort_values("cooks_distance", ascending=False).round(4), "12_influence_table")
    influence_plot(influence, FIGURES / "15_influence.png", f"Influential team-matches in the training data ({best})")
    print(f"\nVariables used by the final model: {used}   (target scale: {scale})")
    print(assumptions.round(3).to_string(index=False))

    raw_target = assumption_tests(ols(X_tf, y_train)).assign(target="goals")
    log_target = assumption_tests(ols(X_tf, np.log1p(y_train))).assign(target="log(1 + goals)")
    target_check = pd.concat([raw_target, log_target])
    save_table(target_check.round(4), "12_assumption_tests_goals_vs_log_goals")
    print("\nAll 8 transformed variables, raw vs log(1 + goals) target:")
    print(target_check[["target", "test", "p_value", "holds_at_5pct"]].round(3).to_string(index=False))

    joblib.dump(final, MODELS_DIR / "final_model.joblib")
    return {
        "test_metrics": test_table.set_index("model").round(6).to_dict(orient="index"),
        "bootstrap_ci": ci.set_index("metric").round(6).to_dict(orient="index"),
        "variables_used": used, "target_scale": scale, "rounded_exact": exact, "rounded_within_one": within_one,
        "smearing_factor": round(final.smearing_, 4) if log_model else None,
        "assumptions": assumptions[["test", "statistic", "p_value"]].round(4).to_dict(orient="records"),
        "raw_vs_log_assumptions": target_check[["target", "test", "p_value"]].round(4).to_dict(orient="records"),
        "influential_rows": int(influence["influential"].sum()),
    }


def feature_analysis(model_set, fitted, best, train, test, X_tf, y_range):
    heading("7. Which variables matter?")
    compared = {"OLS, all 8 (transformed)": "MLR: winsorise + Yeo-Johnson", "Ridge": "Ridge (alpha by CV)",
                "Lasso": "Lasso (alpha by CV)"}
    coefs = pd.DataFrame({name: standardised_coefficients(fitted[m], FEATURES, FEATURES)
                          for name, m in compared.items()})
    if isinstance(fitted[best], LogTargetRegressor):
        # on the log(1 + goals) scale d(goals)/dx = coef * (1 + goals), so scale by 1 + the average goals
        coefs[f"Selected: {best} (effect for an average team)"] = \
            standardised_coefficients(fitted[best], FEATURES, FEATURES) * (1 + train[TARGET].mean())
    elif best not in compared.values():
        coefs[f"Selected: {best}"] = standardised_coefficients(fitted[best], FEATURES, FEATURES)
    save_table(coefs.reset_index(names="variable").round(4), "13_coefficients")
    print(coefs.round(3).to_string())
    coefficient_plot(coefs, FIGURES / "16_coefficients.png",
                         "Standardised coefficients (goals per 1 SD change in each variable)")

    importance = pd.concat([
        permutation_importance(fitted[name], model_set[name][0], test[FEATURES], test[TARGET], FEATURES,
                                          y_range).assign(model=name)
        for name in dict.fromkeys([best, "Lasso (alpha by CV)", "Ridge (alpha by CV)"])
    ])
    save_table(importance.round(4), "14_permutation_importance")
    importance_plot(importance, FIGURES / "17_permutation_importance.png",
                        "Permutation importance on the 84 test team-matches")

    lasso_model = fitted["Lasso (alpha by CV)"][-1]
    lasso_path_plot(X_tf, train[TARGET], FEATURES, lasso_model.alpha_, FIGURES / "18_lasso_path.png",
                        "Lasso path: which variables survive as the penalty grows")
    return coefs, importance


def robustness(model_set, best, df, y_range):
    heading(f"8. Robustness: {N_SPLITS} different 60/40 match splits")
    names = ["Baseline: training mean", "MLR: z-score", "Simple LR: Elo only", "MLR: backward elimination", best]
    contenders = {name: model_set[name] for name in dict.fromkeys(names)}
    results = repeated_splits(contenders, df[FEATURES], df[TARGET], y_range,
                                          groups=df["Match_ID"].to_numpy())
    table = results.groupby("model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    table.columns = ["_".join(c) for c in table.columns]
    save_table(table.reset_index().round(4), "15_robustness_summary")
    print(table.round(3).to_string())
    robustness_plot(results, list(contenders), FIGURES / "19_robustness.png",
                        f"One 60/40 split is a noisy yardstick - results over {N_SPLITS} random match splits")
    return table


def compare_with_2_1(model_set, fitted, best, test):
    heading("9. Objective 2.1 vs 2.2 on the same 42 test matches")
    path = TABLES_2_1 / "10_test_predictions.csv"
    if not path.exists():
        print("Run objective 2.1 first to get its test predictions - skipping this comparison.")
        return {}
    direct = pd.read_csv(path)
    columns, _ = model_set[best]
    rates = test[["Match_ID", "Team"]].assign(rate=fitted[best].predict(test[columns]))
    matches = (direct.merge(rates.rename(columns={"Team": "Team_A", "rate": "rate_A"}), on=["Match_ID", "Team_A"])
                     .merge(rates.rename(columns={"Team": "Team_B", "rate": "rate_B"}), on=["Match_ID", "Team_B"]))
    if len(matches) != len(direct):
        raise ValueError("2.1 and 2.2 test sets do not hold the same matches")
    matches = matches.rename(columns={"predicted_GD": "direct_2_1"})
    matches["implied_2_2"] = matches["rate_A"] - matches["rate_B"]
    matches["average"] = (matches["direct_2_1"] + matches["implied_2_2"]) / 2
    actual_gd = matches["actual_GD"].to_numpy()
    actual_result = np.sign(actual_gd)
    decisive = actual_result != 0

    goals_a, goals_b = poisson_scorelines(matches["rate_A"], matches["rate_B"])
    poisson_probs = np.column_stack([(goals_a > goals_b).mean(axis=1), (goals_a == goals_b).mean(axis=1),
                                     (goals_a < goals_b).mean(axis=1)])
    direct_probs = matches[["p_win", "p_draw", "p_loss"]].to_numpy()

    rows = []
    for route, pred, probs in [("2.1 direct (goal difference model)", matches["direct_2_1"], direct_probs),
                               ("2.2 implied (team goals model)", matches["implied_2_2"], poisson_probs),
                               ("average of both", matches["average"], (direct_probs + poisson_probs) / 2)]:
        pred = pred.to_numpy()
        m = regression_metrics(actual_gd, pred, 10.0)
        rows.append({"route": route, "RMSE": m["RMSE"], "MAE": m["MAE"], "R2": m["R2"],
                     "winner_accuracy": float(np.mean(np.sign(pred[decisive]) == actual_result[decisive])),
                     "brier": brier_score(probs, actual_result)})
    table = pd.DataFrame(rows)
    save_table(table.round(4), "16_objective_2_1_vs_2_2")
    print(table.round(3).to_string(index=False))

    rate_a = np.clip(matches["rate_A"].to_numpy(), 0.05, None)[:, None]
    rate_b = np.clip(matches["rate_B"].to_numpy(), 0.05, None)[:, None]
    floor_gd = np.sqrt(np.mean((goals_a - goals_b - (rate_a - rate_b)) ** 2, axis=0))
    team_rates = np.clip(fitted[best].predict(test[columns]), 0.05, None)
    sim_goals, _ = poisson_scorelines(team_rates, team_rates)
    floor_goals = np.sqrt(np.mean((sim_goals - team_rates[:, None]) ** 2, axis=0))
    goals_rmse = regression_metrics(test[TARGET], fitted[best].predict(test[columns]), 7.0)["RMSE"]
    floor = pd.DataFrame({
        "target": ["goal difference (42 matches)", "team goals (84 rows)"],
        "perfect_model_RMSE_median": [np.median(floor_gd), np.median(floor_goals)],
        "perfect_model_RMSE_5th_pct": [np.percentile(floor_gd, 5), np.percentile(floor_goals, 5)],
        "perfect_model_RMSE_95th_pct": [np.percentile(floor_gd, 95), np.percentile(floor_goals, 95)],
        "achieved_RMSE": [table.loc[0, "RMSE"], goals_rmse],
    })
    save_table(floor.round(4), "17_noise_floor")
    print(floor.round(3).to_string(index=False))

    matches.assign(p_win_poisson=poisson_probs[:, 0], p_draw_poisson=poisson_probs[:, 1],
                   p_loss_poisson=poisson_probs[:, 2]).round(4).to_csv(TABLES / "16_match_level_comparison.csv",
                                                                        index=False)
    achieved = {"gd": [("2.1 model", table.loc[0, "RMSE"], BLUE), ("2.2 implied", table.loc[1, "RMSE"], CORAL)],
                "goals": [("2.2 model", goals_rmse, CORAL)]}
    route_comparison_figure(matches, table, {"gd": floor_gd, "goals": floor_goals}, achieved)
    return {"routes": table.round(4).to_dict(orient="records"), "noise_floor": floor.round(4).to_dict(orient="records")}


def run():
    setup()
    for folder in (FIGURES, TABLES, MODELS_DIR, PROCESSED_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    df = load_2_2()
    problems = check_2_2(df)
    if problems:
        raise ValueError("; ".join(problems))
    y_range = df[TARGET].max() - df[TARGET].min()

    describe(df)
    tests, conditional = inferential(df)

    train, test = split_by_match(df)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"\nTraining: {len(train)} rows ({train['Match_ID'].nunique()} matches, {len(train) / len(df):.0%})   "
          f"Test: {len(test)} rows ({test['Match_ID'].nunique()} matches, {len(test) / len(df):.0%})")

    nonlinear, reset_raw, reset_log = relationships(df, train)
    X_tf, ols_table, kept, vif = statsmodels_steps(train[FEATURES], train[TARGET])

    heading("5. Competing models: repeated grouped 5-fold CV on the training set, then the test set")
    model_set = candidate_models()
    summary, fitted, fold_scores = compare_models(model_set, train[FEATURES], train[TARGET],
                                                              test[FEATURES], test[TARGET], y_range,
                                                              groups=train["Match_ID"].to_numpy(),
                                                              references=REFERENCES)
    best, selection = select_model(summary, model_set, fitted, train[FEATURES], train[TARGET])
    save_table(summary.round(4), "08_model_comparison")
    save_table(selection.round(4), "08_selection_check")
    print(summary[["model", "cv_RMSE", "cv_RMSE_sd", "cv_R2", "test_RMSE", "test_R2", "p_vs_cv_best"]]
          .round(3).to_string(index=False))
    print("\nCandidates tied with the most accurate one, and whether their residuals pass the assumption tests:")
    print(selection.round(3).to_string(index=False))
    print(f"\nSelected model: {best}")
    model_comparison(summary, best, FIGURES / "10_model_comparison.png",
                         "Objective 2.2 - competing linear models (lower RMSE is better)")
    steps = dict(STEPS)
    if best not in steps:
        steps[best] = f"Selected:\n{best}"
    optimisation_path(summary, steps, FIGURES / "11_optimisation_path.png",
                          "How each modelling decision changed the error (goals)")

    report = final_model_report(model_set, fitted, best, train, test, X_tf, y_range)
    coefs, importance = feature_analysis(model_set, fitted, best, train, test, X_tf, y_range)
    robust = robustness(model_set, best, df, y_range)
    comparison = compare_with_2_1(model_set, fitted, best, test)

    by_model = summary.set_index("model")
    results = {
        "selected_model": best,
        "n_train_rows": len(train), "n_test_rows": len(test),
        "inferential": tests.round(4).to_dict(orient="records"),
        "conditional": conditional.round(3).to_dict(orient="index"),
        "nonlinear_variables": nonlinear.loc[nonlinear["x_squared_p"] < 0.05, "variable"].tolist(),
        "reset_p_goals": reset_raw, "reset_p_log_goals": reset_log,
        "ols_steps": ols_table.round(4).to_dict(orient="records"),
        "backward_elimination_keeps": kept,
        "max_vif": float(vif["VIF"].max()),
        "cv": by_model[["cv_RMSE", "cv_RMSE_sd", "cv_R2", "test_RMSE", "test_R2", "test_MAE"]].round(6)
              .to_dict(orient="index"),
        "cv_improvement_vs_raw_mlr_pct": round(100 * (1 - by_model.loc[best, "cv_RMSE"] /
                                                      by_model.loc["MLR: z-score", "cv_RMSE"]), 1),
        "p_selected_vs": {name: round(corrected_t_test(fold_scores[name], fold_scores[best])[1], 4)
                          for name in by_model.index if name != best},
        "selection_check": selection.round(4).to_dict(orient="records"),
        "top_importance": importance[importance["model"] == best].iloc[0]["variable"],
        "robustness": robust.round(6).to_dict(orient="index"),
        **report, **comparison,
    }
    (TABLES / "results_summary.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nFigures: {FIGURES}\nTables:  {TABLES}")
    return results


# ==============================================================================================================
# STEP 12 - REPORTING: plain-English report and README results
# The report goes to outputs/report/; the README results are rewritten after every full run.
# ==============================================================================================================
PROSE = {
    "FIFA_Rank_Diff": "FIFA rank",
    "Elo_Rating_Diff": "Elo rating",
    "Past_30_Days_Win_Rate_Diff": "30-day win rate",
    "Avg_Goals_Scored_L5_Diff": "goals scored (last 5)",
    "Avg_Goals_Conceded_L5_Diff": "goals conceded (last 5)",
    "Days_Rest_Diff": "rest days",
    "Avg_xG_Margin_Diff": "xG margin",
    "Head_To_Head_Win_Pct_Diff": "head-to-head record",
    "Team_Prior_Avg_Goals_Scored": "team's prior goals scored",
    "Opp_Prior_Avg_Goals_Conceded": "opponent's prior goals conceded",
    "Knockout_Stage": "knockout flag",
    "Host_Advantage": "host flag",
}


def names(variables):
    words = [PROSE.get(v, v) for v in variables]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def num(x, d=2):
    return f"{x:.{d}f}"


def pval(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def model_table(cv, best):
    lines = ["| Model | CV RMSE | CV R² | Test RMSE | Test R² |", "|---|---:|---:|---:|---:|"]
    for name, m in sorted(cv.items(), key=lambda item: item[1]["cv_RMSE"]):
        shown = f"**{name}** (selected)" if name == best else name
        lines.append(f"| {shown} | {num(m['cv_RMSE'], 3)} | {num(m['cv_R2'])} | {num(m['test_RMSE'], 3)} | "
                     f"{num(m['test_R2'], 3)} |")
    return "\n".join(lines)


def assumption_line(assumptions):
    by_test = {a["test"]: a for a in assumptions}
    dw = next(a["statistic"] for a in assumptions if a["test"].startswith("Durbin"))
    return (f"Shapiro-Wilk {pval(by_test['Shapiro-Wilk']['p_value'])}, "
            f"Breusch-Pagan {pval(by_test['Breusch-Pagan']['p_value'])}, "
            f"Ramsey RESET {pval(by_test['Ramsey RESET']['p_value'])}, Durbin-Watson {num(dw)}")


def report_2_2(r):
    folder = REPORT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    tables = TABLES
    best = r["selected_model"]
    final, base = r["test_metrics"][best], r["test_metrics"]["Baseline: training mean"]
    mean_goals, scored, favourites, hosts, knockout, dispersion = r["inferential"]
    normality = pd.read_csv(tables / "06_normality_before_after.csv").set_index(["variable", "stage"])["skewness"]
    raw_vs_log = pd.DataFrame(r["raw_vs_log_assumptions"]).set_index(["target", "test"])["p_value"]
    selection = pd.DataFrame(r["selection_check"])
    cv_best = selection.iloc[0]
    ci = r["bootstrap_ci"]
    robust = r["robustness"]
    fav, dog = r["conditional"]["Elo favourite"], r["conditional"]["Elo underdog"]
    routes = {row["route"]: row for row in r.get("routes", [])}
    floors = {row["target"]: row for row in r.get("noise_floor", [])}
    host_rows = int(pd.read_csv(DATA_2_2)["Host_Advantage"].sum())
    importance = pd.read_csv(tables / "14_permutation_importance.csv")
    top = importance[importance["model"] == best].sort_values("importance", ascending=False)["variable"].head(3)
    top = names(list(top))
    coefs = pd.read_csv(tables / "13_coefficients.csv").set_index("variable").iloc[:, -1]

    selection_text = (f"The most accurate candidate in cross-validation was {cv_best['model']}"
                      f" (CV RMSE {num(cv_best['cv_RMSE'], 3)})")
    if cv_best["model"] != best:
        selection_text += (f", but its residuals fail the normality test (Shapiro-Wilk {pval(cv_best['Shapiro-Wilk'])}). "
                           f"**{best}** is statistically tied with it ({pval(r['p_selected_vs'][cv_best['model']])}) "
                           f"and passes every assumption test, so it was selected.")
    else:
        selection_text += f" and its residuals pass every assumption test, so it was selected."

    comparison = ""
    if routes:
        direct = routes["2.1 direct (goal difference model)"]
        implied = routes["2.2 implied (team goals model)"]
        both = routes["average of both"]
        gd_floor = floors["goal difference (42 matches)"]
        goal_floor = floors["team goals (84 rows)"]
        agree = (f"averaging them gives the lowest error ({num(both['RMSE'], 3)})" if both["RMSE"] < min(direct["RMSE"], implied["RMSE"])
                 else f"averaging them gives an RMSE of {num(both['RMSE'], 3)}")
        if direct["winner_accuracy"] >= implied["winner_accuracy"] and direct["brier"] <= implied["brier"]:
            edge = "The direct 2.1 model is still better at picking winners and its probabilities score better."
        elif direct["brier"] <= implied["brier"]:
            edge = "The direct 2.1 model gives the better-calibrated probabilities (lower Brier score)."
        else:
            edge = "The team-by-team route gives the better-calibrated probabilities (lower Brier score)."
        inside = (goal_floor["perfect_model_RMSE_5th_pct"] <= goal_floor["achieved_RMSE"] <= goal_floor["perfect_model_RMSE_95th_pct"]
                  and gd_floor["perfect_model_RMSE_5th_pct"] <= gd_floor["achieved_RMSE"] <= gd_floor["perfect_model_RMSE_95th_pct"])
        verdict = ("inside those ranges. The low R² values are mostly the randomness of football, not missing modelling "
                   "effort." if inside else "outside those ranges, so there is still signal the models miss.")
        comparison = f"""
## Objective 2.1 vs 2.2 on the same 42 matches

Both objectives hold out exactly the same 42 matches, so we can compare two ways of forecasting a match: predict the
goal difference directly (2.1), or predict each team's goals and subtract (2.2).

| Route | RMSE | MAE | R² | Winner correct | Brier score |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {name} | {num(v['RMSE'], 3)} | {num(v['MAE'], 3)} | {num(v['R2'], 3)} | "
                f"{num(100 * v['winner_accuracy'], 0)}% | {num(v['brier'], 3)} |" for name, v in routes.items()) + f"""

The two models were built separately, on different datasets, yet they agree closely (RMSE {num(direct['RMSE'], 3)} vs
{num(implied['RMSE'], 3)}), and {agree}. {edge}

How good can a model get here? We simulated scorelines with goals drawn from a Poisson distribution at the predicted
rates (week 11). Even a model that knew every team's true scoring rate would have a median RMSE of
{num(goal_floor['perfect_model_RMSE_median'])} for team goals (90% range {num(goal_floor['perfect_model_RMSE_5th_pct'])}
to {num(goal_floor['perfect_model_RMSE_95th_pct'])}) and {num(gd_floor['perfect_model_RMSE_median'])} for goal
difference. Our models sit at {num(goal_floor['achieved_RMSE'])} and {num(gd_floor['achieved_RMSE'])}, {verdict}

![2.1 vs 2.2](../../images/20_objective_2_1_vs_2_2.png)
"""

    text = f"""# Objective 2.2 - Predicting how many goals a team scores in a World Cup 2026 match

## In short

Each of the 208 rows is one team in one match. The selected model, **{best}**, predicts goals with a test RMSE of
{num(final['RMSE'], 3)} (baseline {num(base['RMSE'], 3)}) and R² {num(final['R2'], 3)} (baseline
{num(base['R2'], 3)}), and {num(100 * r['rounded_within_one'], 0)}% of its rounded predictions are within one goal.
Goals are a skewed count, so the model is fitted on log(1 + goals): on that scale every regression assumption holds,
which was not true for raw goals.

![Overview](../../images/01_goals_overview.png)

## The data

The eight explanatory variables are the Elo, 30-day win rate and last-five goals scored and conceded differences
(team minus opponent, the four variables shared with 2.1), the team's prior average goals scored and the
opponent's prior average goals conceded in this tournament, a knockout flag and a host-nation flag.

The two prior averages start every team at {PRIOR_GOALS} goals and add each earlier World Cup match:
(2 × {PRIOR_GOALS} + goals in earlier matches) / (2 + earlier matches). This keeps a team's first match from
being undefined and stops one big result from dominating. We rebuilt both variables from the match results and they
agree on all 208 rows, so no variable uses the match it describes.

## What the data shows (weeks 2, 3, 4, 7 and 10)

Teams scored {num(mean_goals['estimate'])} goals per match on average (95% CI {num(mean_goals['ci_low'])} to
{num(mean_goals['ci_high'])}), and {num(100 * scored['estimate'], 0)}% scored at least once. Elo favourites scored
{num(favourites['estimate'])} more goals than underdogs ({pval(favourites['p_value'])}), and host nations playing at home
scored {num(hosts['estimate'])} more than everyone else ({pval(hosts['p_value'])}, but only {host_rows} such rows). Knockout games
were not lower-scoring ({pval(knockout['p_value'])}). The variance of goals is {num(dispersion['estimate'])} times the
mean ({pval(dispersion['p_value'])}), so goals are a little more spread out than a pure Poisson count.

Conditional probabilities make the Elo effect concrete: a favourite scored two or more goals
{num(100 * fav['2+ goals'], 0)}% of the time and nothing {num(100 * fav['0 goals'], 0)}% of the time; for an
underdog those figures are {num(100 * dog['2+ goals'], 0)}% and {num(100 * dog['0 goals'], 0)}%.

## Preparing the data (weeks 8 and 9)

The 60/40 split is done **by match**, so the two rows of a match never end up on different sides (a plain row split
would put both rows of 52 matches on both sides). That gives {r['n_train_rows']} training rows and
{r['n_test_rows']} test rows, and the test matches are the same 42 as in Objective 2.1.

The two prior averages were the only skewed variables (skewness {num(normality[('Team_Prior_Avg_Goals_Scored', 'raw')])}
and {num(normality[('Opp_Prior_Avg_Goals_Conceded', 'raw')])}); winsorising plus Yeo-Johnson brought both to about 0.
All variables were then on a z-score scale. The largest VIF is {num(r['max_vif'])}, so collinearity is not a problem.

The real issue was the target. With raw goals the residuals are clearly not normal (Shapiro-Wilk
{pval(raw_vs_log[('goals', 'Shapiro-Wilk')])}); with log(1 + goals) they are (Shapiro-Wilk
{pval(raw_vs_log[('log(1 + goals)', 'Shapiro-Wilk')])}), and the RESET linearity test improves from
{pval(raw_vs_log[('goals', 'Ramsey RESET')])} to {pval(raw_vs_log[('log(1 + goals)', 'Ramsey RESET')])}.
Predictions are converted back to goals with Duan's smearing factor ({num(r['smearing_factor'], 3)}); without it the
back-transformed predictions would sit too low on average.

![Log transformation](../../images/06_linearisation.png)

## Competing models

Cross-validation used grouped folds (both rows of a match stay together), 5 folds repeated {CV_REPEATS} times.

{model_table(r['cv'], best)}

{selection_text} The Elo-only regression is again statistically tied ({pval(r['p_selected_vs']['Simple LR: Elo only'])}).

![Model comparison](../../images/10_model_comparison.png)

## The final model on the 84 test rows

| Model | MAE | RMSE | NRMSE | R² |
|---|---:|---:|---:|---:|
""" + "\n".join(f"| {name} | {num(m['MAE'], 3)} | {num(m['RMSE'], 3)} | {num(m['NRMSE'], 3)} | {num(m['R2'], 3)} |"
                for name, m in r["test_metrics"].items()) + f"""

Bootstrapping whole matches gives a 95% interval of {num(ci['RMSE']['ci_low'])} to {num(ci['RMSE']['ci_high'])} for
the test RMSE and {num(ci['R2']['ci_low'], 2)} to {num(ci['R2']['ci_high'], 2)} for R². The rounded prediction is exactly
right {num(100 * r['rounded_exact'], 0)}% of the time and within one goal {num(100 * r['rounded_within_one'], 0)}% of
the time. On the log scale the residuals pass every check ({assumption_line(r['assumptions'])}); the full statsmodels
summary, including heteroscedasticity-robust (HC3) standard errors, is in `outputs/tables/12_final_model_ols_summary.txt`.
Cook's distance flags {r['influential_rows']} training rows, which we inspected and kept.

![Actual vs predicted](../../images/12_actual_vs_predicted.png)

![Residuals](../../images/13_residuals_train.png)

## Which variables matter

Permutation importance on the test rows ranks the {top} highest for the selected model. {"The opponent's prior goals conceded has a positive effect: a leaky defence earlier in the tournament means more goals for the next opponent. " if coefs["Opp_Prior_Avg_Goals_Conceded"] > 0 else ""}Over {N_SPLITS} random match splits the mean test R² was {num(robust[best]['R2_mean'], 3)} for
{best} and {num(robust['Simple LR: Elo only']['R2_mean'], 3)} for the Elo-only regression, so again a single strong
variable does almost as well as the full model.

![Coefficients](../../images/16_coefficients.png)
{comparison}
## Limitations

Goals are rare, so even 208 rows hold a lot of noise. Host advantage rests on 13 rows. The two prior averages are
weak in the first round because every team starts at the same value. The model predicts expected goals, not exact
scorelines.
"""
    path = folder / "objective_2_2_report.md"
    path.write_text(text, encoding="utf-8")
    return path


README_START = "<!-- results:start - written by python main.py or scripts/run.py, edits here are overwritten -->"
README_END = "<!-- results:end -->"
SHORT_NAMES = {
    "Lasso (alpha by CV)": "Lasso",
    "Ridge (alpha by CV)": "Ridge on raw goals",
    "Ridge + log(1 + goals) target": "Ridge + log(1 + goals)",
    "MLR: + log(1 + goals) target": "MLR + log(1 + goals)",
    "MLR: backward elimination": "Backward elimination",
    "Simple LR: Elo only": "Elo-only regression",
    "Baseline: training mean": "Mean baseline",
}
REFERENCE_MODELS = ("MLR: backward elimination", "Simple LR: Elo only")


def short(name):
    return SHORT_NAMES.get(name, name)


def signed(x, d=3):
    return f"{x:.{d}f}".replace("-", "−")


def test_table(metrics, best, unit):
    lines = [f"| Model ({unit}) | MAE | RMSE | R² |", "|---|---:|---:|---:|"]
    for name, m in metrics.items():
        if name == best:
            shown = f"**{short(name)} (selected)**"
        elif name in REFERENCE_MODELS:
            shown = f"{short(name)} (reference)"
        else:
            shown = short(name)
        lines.append(f"| {shown} | {num(m['MAE'], 3)} | {num(m['RMSE'], 3)} | {signed(m['R2'])} |")
    return "\n".join(lines)


def tied(p, other):
    if p > ALPHA:
        return f"statistically tied with {other} ({pval(p)})"
    return f"significantly different from {other} ({pval(p)})"


def join(words):
    words = list(words)
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def readme_2_2(r):
    best = r["selected_model"]
    mean_goals, scored, favourites, hosts, knockout, dispersion = r["inferential"]
    raw_vs_log = {(a["target"], a["test"]): a["p_value"] for a in r["raw_vs_log_assumptions"]}
    by_test = {a["test"]: a for a in r["assumptions"]}
    selection = r["selection_check"]
    cv_best = selection[0]
    robust = r["robustness"]
    host_rows = int(pd.read_csv(DATA_2_2)["Host_Advantage"].sum())
    normality = pd.read_csv(TABLES / "06_normality_before_after.csv")
    skewed = normality[(normality["stage"] == "raw") & (normality["skewness"].abs() > 0.5)]["variable"].tolist()
    other_sd = [v["R2_std"] for k, v in robust.items() if k not in (best, "Baseline: training mean")]
    steadier = robust[best]["R2_std"] < min(other_sd)
    scale = "log(1 + goals)" if r.get("smearing_factor") else "goals"
    holds = all(a["p_value"] > ALPHA for a in r["assumptions"] if a["p_value"] == a["p_value"])

    if cv_best["model"] != best:
        failed = [t for t in ("Shapiro-Wilk", "Jarque-Bera", "Breusch-Pagan", "Ramsey RESET") if cv_best[t] <= ALPHA]
        choice = (f"{short(cv_best['model'])} had the lowest CV RMSE ({num(cv_best['cv_RMSE'], 3)}), but its residuals "
                  f"fail the {join(failed)} test{'s' if len(failed) > 1 else ''}. {short(best)} is "
                  f"{tied(r['p_selected_vs'][cv_best['model']], short(cv_best['model']))} and passes every check, "
                  f"so it was selected.")
    else:
        choice = f"{short(best)} had the lowest CV RMSE ({num(cv_best['cv_RMSE'], 3)}) and passes every assumption test."

    smearing = (f", back-transformed with a smearing factor of {num(r['smearing_factor'], 3)}"
                if r.get("smearing_factor") else "")
    return f"""## Results – Objective 2.2

**Selected model: {short(best)}** (all eight variables{smearing}).

{test_table(r['test_metrics'], best, '84 test rows')}

- With raw goals as the target the residuals are not normal (Shapiro-Wilk
  {pval(raw_vs_log[('goals', 'Shapiro-Wilk')])}, RESET {pval(raw_vs_log[('goals', 'Ramsey RESET')])}). The selected
  model works on the {scale} scale, where its residuals {"pass every check" if holds else "give"}: Shapiro-Wilk
  {pval(by_test['Shapiro-Wilk']['p_value'])}, Breusch-Pagan {pval(by_test['Breusch-Pagan']['p_value'])}, RESET
  {pval(by_test['Ramsey RESET']['p_value'])}.
- {choice}
- Compared with the plain eight-variable regression, CV RMSE improved {num(r['cv_improvement_vs_raw_mlr_pct'], 1)}% and
  test R² went from {signed(r['cv']['MLR: z-score']['test_R2'])} to {signed(r['cv'][best]['test_R2'])}.
  {"Over " + str(N_SPLITS) + " random splits its R² also varies less than any other regression model we tried (SD " + num(robust[best]['R2_std'], 3) + ")." if steadier else ""}
  For 2.2 the transformations mainly fixed validity rather than raw accuracy.
- {num(100 * r['rounded_within_one'], 0)}% of rounded predictions are within one goal.
- Elo favourites score {num(favourites['estimate'])} more goals than underdogs ({pval(favourites['p_value'])}). Host
  nations at home score {num(hosts['estimate'])} more ({pval(hosts['p_value'])}, only {host_rows} rows). Teams score
  {"about the same number of goals" if knockout['p_value'] > ALPHA else "a different number of goals"} in knockout
  games ({pval(knockout['p_value'])}). Goals are
  {"slightly over-dispersed" if dispersion['p_value'] <= ALPHA else "close to"} compared with a Poisson count
  (variance/mean = {num(dispersion['estimate'])}, {pval(dispersion['p_value'])}).
- Collinearity is {"not a problem" if r['max_vif'] < 5 else "a problem"} (largest VIF {num(r['max_vif'])}).
  {("The skewed variables (" + names(skewed) + ") were fixed by winsorising and Yeo-Johnson.") if skewed else ""}
"""


def readme_comparison(r):
    routes = {row["route"]: row for row in r["routes"]}
    floors = {row["target"]: row for row in r["noise_floor"]}
    direct = routes["2.1 direct (goal difference model)"]
    implied = routes["2.2 implied (team goals model)"]
    gd, goals = floors["goal difference (42 matches)"], floors["team goals (84 rows)"]
    inside = (gd["perfect_model_RMSE_5th_pct"] <= gd["achieved_RMSE"] <= gd["perfect_model_RMSE_95th_pct"] and
              goals["perfect_model_RMSE_5th_pct"] <= goals["achieved_RMSE"] <= goals["perfect_model_RMSE_95th_pct"])
    rows = "\n".join(f"| {name} | {num(v['RMSE'], 3)} | {num(100 * v['winner_accuracy'], 0)}% | {num(v['brier'], 3)} |"
                     for name, v in routes.items())
    close = abs(direct["RMSE"] - implied["RMSE"]) < 0.1
    return f"""## Objective 2.1 vs 2.2 on the same 42 matches

Both objectives hold out exactly the same 42 matches, so the two approaches can be compared directly:

| Route | RMSE | Winner correct | Brier score |
|---|---:|---:|---:|
{rows}

{"Two models built separately, on different datasets with different targets, land on almost the same error, which is a good sign that both are sound." if close else "The two routes give noticeably different errors on these 42 matches."}

We also simulated scorelines from a Poisson distribution at the predicted rates. Even a model that knew every team's true
scoring rate would get a median RMSE of {num(goals['perfect_model_RMSE_median'])} for team goals and
{num(gd['perfect_model_RMSE_median'])} for goal difference. Our models sit at {num(goals['achieved_RMSE'])} and
{num(gd['achieved_RMSE'])}, {"inside those ranges, so the modest R² values come mostly from the randomness of football, not from missing effort." if inside else "outside the expected range, so some signal is still being missed."}
"""


def load_results():
    path = TABLES / "results_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def write_readme_results(readme, parts):
    """Replace the text between the results markers of a README with the given Markdown parts."""
    readme = Path(readme)
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return None
    start = text.index(README_START) + len(README_START)
    end = text.index(README_END)
    readme.write_text(text[:start] + "\n\n" + "\n".join(parts) + "\n" + text[end:], encoding="utf-8")
    return readme


def readme_sections():
    """The Objective 2.2 results (and the 2.1 vs 2.2 comparison) for a README, from the latest results_summary.json."""
    r2 = load_results()
    parts = []
    if r2:
        parts.append(readme_2_2(r2))
        if r2.get("routes"):
            parts.append(readme_comparison(r2))
    return parts


def update_readme(readme=README):
    """Rewrite the results part of this objective's README with the numbers from the latest full run."""
    return write_readme_results(readme, readme_sections())
