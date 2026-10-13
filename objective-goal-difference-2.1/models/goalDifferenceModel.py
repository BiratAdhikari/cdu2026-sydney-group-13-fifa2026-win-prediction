"""Objective 2.1 - predict the goal difference (Team A - Team B) of each World Cup 2026 match.

Everything Objective 2.1 needs is in this one file, in the order the analysis runs:

    Settings
    STEP 1   Data wrangling      load the raw CSV, rename the target, turn "25.0%" into 25.0, parse dates
    STEP 2   Data checks         check the dataset against the brief, save the cleaned file
    STEP 3   Data preparation    60/40 train/test split, saved to data/processed/
    STEP 4   Descriptive and inferential statistics
    STEP 5   Preprocessing       winsorise + Yeo-Johnson + z-scores, backward elimination
    STEP 6   Model builders
    STEP 7   Evaluation
    STEP 8   Diagnostics
    STEP 9   Experiments         cross-validation and model selection
    STEP 10  Visualisation
    STEP 11  The Objective 2.1 analysis (run() calls every step in order)
    STEP 12  Reporting           plain-English report and the README results

Run it with:  python scripts/run.py        (from this folder)   or   python main.py  (from the project root)
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
OBJECTIVE_DIR = Path(__file__).resolve().parents[1]          # objective-goal-difference-2.1/
PROJECT_DIR = OBJECTIVE_DIR.parent
DATA_DIR = OBJECTIVE_DIR / "data"
DATA_2_1 = DATA_DIR / "raw" / "raw.csv"                       # raw dataset, never modified
PROCESSED_DIR = DATA_DIR / "processed"                        # cleaned data and the exact 60/40 train/test files
FIGURES = OBJECTIVE_DIR / "images"                            # all figures
OUTPUTS_DIR = OBJECTIVE_DIR / "outputs"
TABLES = OUTPUTS_DIR / "tables"                               # all tables, statsmodels summaries, results_summary.json
REPORT_DIR = OUTPUTS_DIR / "report"                           # plain-English report
CHECKS_DIR = OUTPUTS_DIR / "data_checks"                      # dataset checks
MODELS_DIR = OBJECTIVE_DIR / "models"                         # this code + the saved final model
README = OBJECTIVE_DIR / "README.md"

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


RAW_TARGET_2_1 = "Goal_Difference (Target)"
TARGET_2_1 = "Goal_Difference"
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

LABELS = {
    "FIFA_Rank_Diff": "FIFA rank diff",
    "Elo_Rating_Diff": "Elo rating diff",
    "Past_30_Days_Win_Rate_Diff": "30-day win rate diff",
    "Avg_Goals_Scored_L5_Diff": "Goals scored (last 5) diff",
    "Avg_Goals_Conceded_L5_Diff": "Goals conceded (last 5) diff",
    "Days_Rest_Diff": "Rest days diff",
    "Avg_xG_Margin_Diff": "xG margin diff",
    "Head_To_Head_Win_Pct_Diff": "Head-to-head win % diff",
    "Goal_Difference": "Goal difference (Team A - Team B)",
}


# ==============================================================================================================
# STEP 1 - DATA WRANGLING: load and clean the raw data
# Reads data/raw.csv and fixes column names and types. The raw file itself is never changed.
# ==============================================================================================================
def load_2_1(path=DATA_2_1):
    df = pd.read_csv(path)
    df = df.rename(columns={RAW_TARGET_2_1: TARGET_2_1})
    df["Head_To_Head_Win_Pct_Diff"] = df["Head_To_Head_Win_Pct_Diff"].astype(str).str.rstrip("%").astype(float)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ==============================================================================================================
# STEP 2 - DATA CHECKS: verify the dataset against the assessment brief
# Results go to outputs/data_checks/data_verification.csv.
# ==============================================================================================================
def check_2_1(df):
    problems = []
    if len(df) != 104:
        problems.append(f"expected 104 matches, found {len(df)}")
    if len(FEATURES_2_1) != 8:
        problems.append("2.1 needs exactly 8 explanatory variables")
    if df[FEATURES_2_1 + [TARGET_2_1]].isna().any().any():
        problems.append("missing values")
    if df["Match_ID"].duplicated().any():
        problems.append("duplicate Match_ID")
    if not (df["Team_A_Goals"] - df["Team_B_Goals"]).eq(df[TARGET_2_1]).all():
        problems.append("goal difference is not Team_A_Goals - Team_B_Goals")
    return problems


def run_checks():
    """Check the 2.1 dataset against the brief and save the cleaned 2.1 file."""
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matches = load_2_1()
    problems = check_2_1(matches)
    if problems:
        raise ValueError("; ".join(problems))
    matches.to_csv(PROCESSED_DIR / "cleaned_dataset.csv", index=False)

    gd_ok = int((matches["Team_A_Goals"] - matches["Team_B_Goals"]).eq(matches[TARGET_2_1]).sum())

    checks = [
        ("2.1 rows (one per match)", len(matches), "104"),
        ("2.1 explanatory variables", len(FEATURES_2_1), "8"),
        ("2.1 missing values", int(matches[FEATURES_2_1 + [TARGET_2_1]].isna().sum().sum()), "0"),
        ("2.1 rows where goal difference = Team A goals - Team B goals", gd_ok, "104"),
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
def split_2_1(df):
    return train_test_split(df[FEATURES_2_1], df[TARGET_2_1], test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)


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
# STEP 11 - OBJECTIVE 2.1 ANALYSIS: GOAL DIFFERENCE
# run() is the full pipeline: wrangle + check -> describe -> inferential tests -> 60/40 split ->
# relationships -> OLS steps -> competing models -> final model -> variables -> robustness.
# ==============================================================================================================
FEATURES = FEATURES_2_1
TARGET = TARGET_2_1
NO_RANK = [f for f in FEATURES if f != "FIFA_Rank_Diff"]
RESULT_NAMES = {1: "Team A win", 0: "Draw", -1: "Team B win"}
RESULT_COLOURS = {"Team A win": GREEN, "Draw": GOLD, "Team B win": CORAL}
REFERENCES = ("Baseline: training mean", "Simple LR: Elo only", "MLR: backward elimination")
STEPS = {
    "MLR: z-score": "MLR, 8 variables\n(z-score only)",
    "MLR: winsorise + Yeo-Johnson": "+ winsorise\n+ Yeo-Johnson",
    "MLR: FIFA rank dropped": "+ drop collinear\nFIFA rank",
    "Ridge (alpha by CV)": "Ridge\n(L2 penalty)",
    "Lasso (alpha by CV)": "Lasso\n(L1 penalty)",
}


def save_table(df, name):
    df.to_csv(TABLES / f"{name}.csv", index=False)


def heading(text):
    print("\n" + "=" * 72 + f"\n2.1  {text}\n" + "=" * 72)


def results_of(goal_difference):
    return pd.Series(np.sign(goal_difference)).map(RESULT_NAMES).to_numpy()


def candidate_models():
    return {
        "Baseline: training mean": (FEATURES, baseline()),
        "Simple LR: Elo only": (["Elo_Rating_Diff"], simple_lr()),
        "MLR: z-score": (FEATURES, zscore_mlr()),
        "MLR: winsorise + Yeo-Johnson": (FEATURES, transformed_mlr(FEATURES)),
        "MLR: FIFA rank dropped": (NO_RANK, transformed_mlr(NO_RANK)),
        "MLR: backward elimination": (NO_RANK, backward_elimination(NO_RANK)),
        "Ridge (alpha by CV)": (FEATURES, ridge(FEATURES)),
        "Lasso (alpha by CV)": (FEATURES, lasso(FEATURES)),
    }


def overview_figure(df):
    gd = df[TARGET]
    result = results_of(gd)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(gd, bins=np.arange(gd.min() - 0.5, gd.max() + 1.5), density=True, color=BLUE, edgecolor="white")
    grid = np.linspace(gd.min() - 1, gd.max() + 1, 200)
    ax.plot(grid, st.norm.pdf(grid, gd.mean(), gd.std()), color=CORAL, lw=2, label="normal curve")
    ax.axvline(gd.mean(), color="black", ls="--", lw=1, label=f"mean = {gd.mean():.2f}")
    ax.set(title="Goal difference across the 104 matches", xlabel="Goal difference (Team A - Team B)",
           ylabel="Share of matches")
    ax.text(0.97, 0.75, f"sd = {gd.std():.2f}\nskew = {st.skew(gd):.2f}", transform=ax.transAxes, ha="right",
            bbox=dict(boxstyle="round", fc="white", ec=GREY))
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    counts = pd.Series(result).value_counts().reindex(RESULT_NAMES.values())
    bars = ax.bar(counts.index, counts.values, color=[RESULT_COLOURS[r] for r in counts.index])
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 1, f"{n} ({n / len(gd):.0%})", ha="center")
    ax.set(title="Match results", ylabel="Matches")
    ax.margins(y=0.15)

    ax = axes[1, 0]
    status = np.where(df["Elo_Rating_Diff"] > 0, "Team A has the\nhigher Elo", "Team A has the\nlower Elo")
    shares = pd.crosstab(status, result, normalize="index")[list(RESULT_NAMES.values())]
    left = np.zeros(len(shares))
    for col in shares.columns:
        ax.barh(shares.index, shares[col], left=left, color=RESULT_COLOURS[col], label=col)
        for i, value in enumerate(shares[col]):
            if value > 0.06:
                ax.text(left[i] + value / 2, i, f"{value:.0%}", ha="center", va="center", color="white",
                        fontweight="bold")
        left += shares[col].to_numpy()
    ax.set(title="P(result | which team is the Elo favourite)", xlabel="Conditional probability", xlim=(0, 1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    ax = axes[1, 1]
    stage = np.where(df["Stage"].str.startswith("Group"), "Group stage", "Knockout")
    plot_df = pd.DataFrame({"stage": stage, "margin": gd.abs()})
    sns.boxplot(data=plot_df, x="stage", y="margin", ax=ax, color="#a9cce3", showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black"))
    sns.stripplot(data=plot_df, x="stage", y="margin", ax=ax, color=BLUE, alpha=0.5, jitter=0.25, size=4)
    ax.set(title="Winning margin by stage (diamond = mean)", xlabel="", ylabel="|Goal difference|")

    fig.suptitle("Objective 2.1 - what the target looks like", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, FIGURES / "01_goal_difference_overview.png")


def describe(df):
    heading("1. Descriptive statistics (weeks 2 and 7)")
    table = descriptive_table(df, [TARGET] + FEATURES)
    save_table(table, "01_descriptive_statistics")
    print(table[["variable", "mean", "median", "std", "min", "max", "iqr", "skewness"]].to_string(index=False))
    overview_figure(df)
    histograms(df, FEATURES, FIGURES / "02_variable_histograms.png",
                   "Distribution of the eight explanatory variables (104 matches)")
    longest = df.loc[df["Days_Rest_Diff"].idxmin()]
    standardised_boxplot(df, FEATURES, FIGURES / "03_outliers_boxplot.png",
                             "Outlier screening on a common z-score scale",
                             notes=[("Days_Rest_Diff", longest["Days_Rest_Diff"],
                                     f"Match {longest['Match_ID']}: Uruguay had not played for 76 days")])


def inferential(df):
    heading("2. Inferential statistics (weeks 3, 4 and 10)")
    gd = df[TARGET]
    rows = []

    mean, low, high = mean_ci(gd)
    rows.append({"question": "Average goal difference (Team A - Team B)", "method": "95% confidence interval (z)",
                 "estimate": mean, "ci_low": low, "ci_high": high})

    t, p = one_sample_t(gd, 0, alternative="greater")
    rows.append({"question": "Does Team A score more than Team B on average?",
                 "method": "one-sample t-test, H0: mean = 0, H1: mean > 0", "estimate": mean, "statistic": t,
                 "p_value": p, "conclusion": conclusion(p)})

    elo_fit = ols(df[["Elo_Rating_Diff"]], gd)
    p = elo_fit.pvalues["const"]
    rows.append({"question": "Is there still a Team A advantage once Elo is accounted for?",
                 "method": "t-test on the intercept of GD ~ Elo diff", "estimate": elo_fit.params["const"],
                 "statistic": elo_fit.tvalues["const"], "p_value": p, "conclusion": conclusion(p)})

    favourite_gd = np.where(df["Elo_Rating_Diff"] > 0, gd, -gd)
    share, low, high = proportion_ci(int((favourite_gd > 0).sum()), int((favourite_gd != 0).sum()))
    rows.append({"question": "How often does the higher-Elo team win a match that has a winner?",
                 "method": "95% confidence interval for a proportion", "estimate": share, "ci_low": low,
                 "ci_high": high})

    group = gd[df["Stage"].str.startswith("Group")].abs()
    knockout = gd[~df["Stage"].str.startswith("Group")].abs()
    t, p = welch_t(group, knockout, alternative="greater")
    rows.append({"question": "Are knockout matches closer than group matches?",
                 "method": "Welch two-sample t-test on |GD|, H1: group > knockout",
                 "estimate": group.mean() - knockout.mean(), "statistic": t, "p_value": p,
                 "conclusion": conclusion(p)})

    table = pd.DataFrame(rows)
    save_table(table.round(4), "02_inferential_tests")
    print(table[["question", "estimate", "ci_low", "ci_high", "p_value", "conclusion"]].round(3).to_string(index=False))

    status = np.where(df["Elo_Rating_Diff"] > 0, "Team A higher Elo", "Team A lower Elo")
    two_way = pd.crosstab(status, results_of(gd), margins=True)
    conditional = pd.crosstab(status, results_of(gd), normalize="index").round(3)
    save_table(two_way.reset_index(names="Elo status"), "03_two_way_table")
    save_table(conditional.reset_index(names="Elo status"), "03_conditional_probabilities")
    print("\nP(result | Elo status):")
    print(conditional)
    return table


def relationships(df, X_train, y_train):
    heading("3. Correlation, linearity and collinearity (weeks 8 and 9)")
    r = df["FIFA_Rank_Diff"].corr(df["Elo_Rating_Diff"])
    heatmap(df, FEATURES + [TARGET], FIGURES / "04_correlation_heatmap.png",
                f"Correlation matrix - FIFA rank and Elo are almost the same signal (r = {r:.2f})")
    linearity_grid(df, FEATURES, TARGET, FIGURES / "05_linearity_check.png",
                       "Is each relationship linear? Straight-line fit vs flexible LOWESS curve")
    nonlinear = nonlinearity_tests(X_train, y_train)
    save_table(nonlinear.round(4), "04_nonlinearity_tests")
    print(nonlinear.round(3).to_string(index=False))

    strongest = df[FEATURES].corrwith(df[TARGET]).abs().sort_values(ascending=False).index[:3].tolist()
    pairplot(df.assign(Result=results_of(df[TARGET])), strongest + [TARGET], "Result",
                 FIGURES / "06_pairplot.png", "The three strongest predictors, coloured by result",
                 palette=RESULT_COLOURS)
    return nonlinear


def statsmodels_steps(X_train, y_train):
    heading("4. Class-style OLS steps on the training set (week 9, statsmodels)")
    X_tf = pd.DataFrame(transformer(FEATURES).fit(X_train).transform(X_train), columns=FEATURES, index=X_train.index)
    X_z = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=FEATURES, index=X_train.index)
    kept = [NO_RANK[i] for i in BackwardElimination().fit(X_tf[NO_RANK], y_train).keep_]

    steps = {
        "1. Raw variables": (X_train, y_train),
        "2. Z-score standardised": (X_z, y_train),
        "3. Winsorised + Yeo-Johnson": (X_tf, y_train),
        "4. FIFA rank dropped (collinearity)": (X_tf[NO_RANK], y_train),
        "5. Backward elimination (p < 0.05)": (X_tf[kept], y_train),
    }
    table = ols_steps(steps, TABLES)
    save_table(table.round(4), "05_ols_steps_summary")
    print(table.round(3).to_string(index=False))
    print("Backward elimination keeps:", kept)

    normality = pd.concat([normality_table(X_train, FEATURES, "raw"),
                           normality_table(X_tf, FEATURES, "winsorised + Yeo-Johnson")])
    save_table(normality.round(4), "06_normality_before_after")
    transformation_plot(X_train, X_tf, FEATURES, "Days_Rest_Diff", FIGURES / "08_transformation.png",
                            "Making the explanatory variables more Gaussian (training data)")

    vif_all, vif_fixed = vif_table(X_tf), vif_table(X_tf[NO_RANK])
    save_table(vif_all.merge(vif_fixed, on="variable", how="left", suffixes=("_all_8", "_rank_dropped")).round(3),
               "07_vif")
    vif_bars(vif_all, vif_fixed, FIGURES / "07_vif.png",
                 "Collinearity check: dropping FIFA rank brings every VIF under 5")
    return X_tf, table, kept


def final_model_report(model_set, fitted, best, summary, X_train, X_test, y_train, y_test, df, y_range):
    heading(f"6. Final model on the 40% test set: {best}")
    columns, _ = model_set[best]
    final = fitted[best]
    pred_test, pred_train = final.predict(X_test[columns]), final.predict(X_train[columns])

    rows = []
    for name in dict.fromkeys([best, "MLR: backward elimination", "Simple LR: Elo only", "Baseline: training mean"]):
        cols, _ = model_set[name]
        rows.append({"model": name, **regression_metrics(y_test, fitted[name].predict(X_test[cols]), y_range)})
    test_table = pd.DataFrame(rows)
    save_table(test_table.round(4), "09_final_vs_benchmarks_test")
    for row in test_table.itertuples():
        print(f"\n{row.model}\nMAE:  {row.MAE:.3f}\nMSE:  {row.MSE:.3f}\nRMSE: {row.RMSE:.3f}\n"
              f"RMSE (normalised): {row.NRMSE:.3f}\nR^2:  {row.R2:.3f}")
    ci = bootstrap_ci(y_test, pred_test, y_range)
    save_table(ci.round(4), "09_final_model_bootstrap_ci")

    test_rows = df.loc[X_test.index]
    names = (test_rows["Team_A"] + " v " + test_rows["Team_B"]).to_numpy()
    cv_rmse = summary.loc[summary["model"] == best, "cv_RMSE"].iloc[0]
    probs = normal_outcome_probabilities(pred_test, cv_rmse)
    actual_result = np.sign(y_test.to_numpy())
    predictions = pd.DataFrame({
        "Match_ID": test_rows["Match_ID"].to_numpy(), "Stage": test_rows["Stage"].to_numpy(),
        "Team_A": test_rows["Team_A"].to_numpy(), "Team_B": test_rows["Team_B"].to_numpy(),
        "actual_GD": y_test.to_numpy(), "predicted_GD": pred_test.round(3),
        "error": (y_test.to_numpy() - pred_test).round(3),
        "p_win": probs[:, 0], "p_draw": probs[:, 1], "p_loss": probs[:, 2],
    }).sort_values("Match_ID")
    save_table(predictions, "10_test_predictions")

    predicted_result = outcome(pred_test)
    confusion = pd.crosstab(pd.Series(results_of(actual_result), name="actual"),
                            pd.Series(results_of(predicted_result), name="predicted"))
    save_table(confusion.reset_index(), "11_result_confusion_matrix")
    decisive = actual_result != 0
    accuracy = float(np.mean(predicted_result == actual_result))
    winner_accuracy = float(np.mean(np.sign(pred_test[decisive]) == actual_result[decisive]))
    shares = pd.Series(np.sign(y_train)).value_counts(normalize=True)
    baseline_probs = np.tile([shares.get(1, 0), shares.get(0, 0), shares.get(-1, 0)], (len(y_test), 1))
    brier_model = brier_score(probs, actual_result)
    brier_base = brier_score(baseline_probs, actual_result)
    print(f"\nResult accuracy (win/draw/loss from the rounded prediction): {accuracy:.1%}")
    print(f"Winner picked correctly in decisive test matches: {winner_accuracy:.1%} of {decisive.sum()}")
    print(f"Brier score: model {brier_model:.3f} vs baseline {brier_base:.3f} (lower is better)")

    actual_vs_predicted(y_test, pred_test, names, FIGURES / "11_actual_vs_predicted.png",
                            f"{best}: actual vs predicted goal difference (42 test matches)", jitter=0.12)
    residual_diagnostics(pred_train, y_train - pred_train, FIGURES / "12_residuals_train.png",
                             f"Residual diagnostics, training set ({best})")
    residual_diagnostics(pred_test, y_test - pred_test, FIGURES / "13_residuals_test.png",
                             f"Residual diagnostics, test set ({best})")
    outcome_probability_bars(probs, actual_result, names, FIGURES / "19_outcome_probabilities.png",
                                 "Simulated result probabilities for the 42 test matches (10,000 runs each)",
                                 list(RESULT_NAMES.values()), RESULT_COLOURS)

    used = used_variables(final, columns)
    post_fit = refit_ols(final, columns, X_train, y_train)
    (TABLES / "12_final_model_ols_summary.txt").write_text(
        f"OLS refit on the variables used by {best} (training set, transformed scale)\n\n{post_fit.summary()}\n")
    assumptions = assumption_tests(post_fit)
    save_table(assumptions.round(4), "12_assumption_tests")
    train_names = df.loc[X_train.index, "Team_A"] + " v " + df.loc[X_train.index, "Team_B"]
    influence = influence_table(post_fit, train_names.to_numpy())
    save_table(influence.sort_values("cooks_distance", ascending=False).round(4), "12_influence_table")
    influence_plot(influence, FIGURES / "14_influence.png", f"Influential matches in the training data ({best})")
    print("\nVariables used by the final model:", used)
    print(assumptions.round(3).to_string(index=False))

    joblib.dump(final, MODELS_DIR / "final_model.joblib")
    return {
        "test_metrics": test_table.set_index("model").round(6).to_dict(orient="index"),
        "bootstrap_ci": ci.set_index("metric").round(6).to_dict(orient="index"),
        "variables_used": used, "result_accuracy": accuracy, "winner_accuracy": winner_accuracy,
        "decisive_test_matches": int(decisive.sum()), "winners_correct": int(round(winner_accuracy * decisive.sum())),
        "brier_model": brier_model, "brier_baseline": brier_base,
        "assumptions": assumptions[["test", "statistic", "p_value"]].round(4).to_dict(orient="records"),
        "influential_matches": int(influence["influential"].sum()),
        "biggest_misses": predictions.reindex(predictions["error"].abs().sort_values(ascending=False).index)
                                     .head(3)[["Team_A", "Team_B", "actual_GD", "predicted_GD"]]
                                     .round(2).to_dict(orient="records"),
    }


def feature_analysis(model_set, fitted, best, X_test, y_test, X_tf, y_train, y_range):
    heading("7. Which variables matter?")
    compared = {"OLS, all 8 (transformed)": "MLR: winsorise + Yeo-Johnson", "Ridge": "Ridge (alpha by CV)",
                "Lasso": "Lasso (alpha by CV)"}
    if best not in compared.values():
        compared[f"Selected: {best}"] = best
    coefs = pd.DataFrame({name: standardised_coefficients(fitted[m], model_set[m][0], FEATURES)
                          for name, m in compared.items()})
    save_table(coefs.reset_index(names="variable").round(4), "13_coefficients")
    print(coefs.round(3).to_string())
    coefficient_plot(coefs, FIGURES / "15_coefficients.png",
                         "Standardised coefficients: Ridge splits the weight between Elo and FIFA rank, Lasso keeps Elo")

    importance = pd.concat([
        permutation_importance(fitted[name], model_set[name][0], X_test, y_test, FEATURES, y_range)
        .assign(model=name)
        for name in dict.fromkeys([best, "Lasso (alpha by CV)", "Ridge (alpha by CV)"])
    ])
    save_table(importance.round(4), "14_permutation_importance")
    importance_plot(importance, FIGURES / "16_permutation_importance.png",
                        "Permutation importance on the 42 test matches")

    lasso_model = fitted["Lasso (alpha by CV)"][-1]
    lasso_path_plot(X_tf, y_train, FEATURES, lasso_model.alpha_, FIGURES / "17_lasso_path.png",
                        "Lasso path: which variables survive as the penalty grows")
    return coefs, importance


def robustness(model_set, best, df, y_range):
    heading(f"8. Robustness: {N_SPLITS} different 60/40 splits")
    names = ["Baseline: training mean", "MLR: z-score", "Simple LR: Elo only", "MLR: backward elimination", best]
    contenders = {name: model_set[name] for name in dict.fromkeys(names)}
    results = repeated_splits(contenders, df[FEATURES], df[TARGET], y_range)
    table = results.groupby("model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    table.columns = ["_".join(c) for c in table.columns]
    save_table(table.reset_index().round(4), "15_robustness_summary")
    print(table.round(3).to_string())
    robustness_plot(results, list(contenders), FIGURES / "18_robustness.png",
                        f"One 60/40 split is a noisy yardstick - results over {N_SPLITS} random splits")
    return table


def run():
    setup()
    for folder in (FIGURES, TABLES, MODELS_DIR, PROCESSED_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    df = load_2_1()
    problems = check_2_1(df)
    if problems:
        raise ValueError("; ".join(problems))
    y_range = df[TARGET].max() - df[TARGET].min()

    describe(df)
    tests = inferential(df)

    X_train, X_test, y_train, y_test = split_2_1(df)
    df.loc[X_train.index].to_csv(PROCESSED_DIR / "train.csv", index=False)
    df.loc[X_test.index].to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"\nTraining matches: {len(X_train)} (60%)   Test matches: {len(X_test)} (40%)")

    nonlinear = relationships(df, X_train, y_train)
    X_tf, ols_table, kept = statsmodels_steps(X_train, y_train)

    heading("5. Competing models: repeated 5-fold CV on the training set, then the test set")
    model_set = candidate_models()
    summary, fitted, fold_scores = compare_models(model_set, X_train, y_train, X_test, y_test, y_range,
                                                              references=REFERENCES)
    best, selection = select_model(summary, model_set, fitted, X_train, y_train)
    save_table(summary.round(4), "08_model_comparison")
    save_table(selection.round(4), "08_selection_check")
    print(summary[["model", "cv_RMSE", "cv_RMSE_sd", "cv_R2", "test_RMSE", "test_R2", "p_vs_cv_best"]]
          .round(3).to_string(index=False))
    print("\nCandidates tied with the most accurate one, and whether their residuals pass the assumption tests:")
    print(selection.round(3).to_string(index=False))
    print(f"\nSelected model: {best}")
    model_comparison(summary, best, FIGURES / "09_model_comparison.png",
                         "Objective 2.1 - competing linear models (lower RMSE is better)")
    steps = dict(STEPS)
    if best not in steps:
        steps[best] = f"Selected:\n{best}"
    optimisation_path(summary, steps, FIGURES / "10_optimisation_path.png",
                          "How each modelling decision changed the error (goal difference)")

    report = final_model_report(model_set, fitted, best, summary, X_train, X_test, y_train, y_test, df, y_range)
    coefs, importance = feature_analysis(model_set, fitted, best, X_test, y_test, X_tf, y_train, y_range)
    robust = robustness(model_set, best, df, y_range)

    by_model = summary.set_index("model")
    results = {
        "selected_model": best,
        "n_train": len(X_train), "n_test": len(X_test),
        "inferential": tests.round(4).to_dict(orient="records"),
        "nonlinear_variables": nonlinear.loc[nonlinear["x_squared_p"] < 0.05, "variable"].tolist(),
        "ols_steps": ols_table.round(4).to_dict(orient="records"),
        "backward_elimination_keeps": kept,
        "cv": by_model[["cv_RMSE", "cv_RMSE_sd", "cv_R2", "test_RMSE", "test_R2", "test_MAE"]].round(6)
              .to_dict(orient="index"),
        "cv_improvement_vs_raw_mlr_pct": round(100 * (1 - by_model.loc[best, "cv_RMSE"] /
                                                      by_model.loc["MLR: z-score", "cv_RMSE"]), 1),
        "p_selected_vs": {name: round(corrected_t_test(fold_scores[name], fold_scores[best])[1], 4)
                          for name in by_model.index if name != best},
        "lasso_keeps": [v for v in FEATURES if abs(coefs.loc[v, "Lasso"]) > 1e-8],
        "top_importance": importance[importance["model"] == best].iloc[0]["variable"],
        "robustness": robust.round(6).to_dict(orient="index"),
        **report,
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


def report_2_1(r):
    folder = REPORT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    tables = TABLES
    best = r["selected_model"]
    final, base = r["test_metrics"][best], r["test_metrics"]["Baseline: training mean"]
    mean_gd, team_a, after_elo, favourite, knockout = r["inferential"]
    steps = r["ols_steps"]
    normality = pd.read_csv(tables / "06_normality_before_after.csv")
    rest = normality[normality["variable"] == "Days_Rest_Diff"].set_index("stage")["skewness"]
    ci = r["bootstrap_ci"]
    robust = r["robustness"]
    raw_cv = r["cv"]["MLR: z-score"]
    misses = r["biggest_misses"]
    kept = names(r["lasso_keeps"])
    nonlinear = ", ".join(r["nonlinear_variables"]) or "none of them"

    text = f"""# Objective 2.1 - Predicting the goal difference of World Cup 2026 matches

## In short

We predicted the goal difference (Team A minus Team B) of all 104 World Cup 2026 matches from eight variables known
before kick-off. The selected model, **{best}**, beat the mean baseline on the 42 held-out test matches (RMSE
{num(final['RMSE'])} vs {num(base['RMSE'])} goals, R² {num(final['R2'], 3)} vs {num(base['R2'], 3)}) and picked the
winner in **{r['winners_correct']} of {r['decisive_test_matches']}** test matches that had one. Most of the signal
comes from the Elo rating difference: Lasso keeps only the {kept} differences, and a one-variable Elo model performs
almost as well.

![Overview](../../images/01_goal_difference_overview.png)

## The data

Each row is one match. The eight explanatory variables are Team A minus Team B differences in FIFA rank, Elo rating,
30-day win rate, goals scored and conceded over the last five matches, rest days, expected-goal margin and
head-to-head win percentage. None of them uses information from the match itself. The data was compiled from the
FIFA website, The Stats Don't Lie and FBref, and it passes every check in `outputs/data_checks/data_verification.csv`.
The extreme rest-days value of -70 in match 13 is genuine: Uruguay had not played since 31 March, 76 days earlier,
while Saudi Arabia had played six days before.

## What the data shows (weeks 2, 3, 4, 7 and 10)

The average goal difference is {num(mean_gd['estimate'])} (95% CI {num(mean_gd['ci_low'])} to
{num(mean_gd['ci_high'])}), and a one-sample t-test says Team A really does better on average ({pval(team_a['p_value'])}).
That advantage is not about being listed first: once Elo is in the model the intercept drops to
{num(after_elo['estimate'])} and is no longer significant ({pval(after_elo['p_value'])}). Team A is simply the stronger
side more often. The team with the higher Elo won {num(100 * favourite['estimate'], 0)}% of matches that had a winner
(95% CI {num(100 * favourite['ci_low'], 0)}-{num(100 * favourite['ci_high'], 0)}%). Knockout games were not
significantly closer than group games (Welch t-test {pval(knockout['p_value'])}).

Most variables are close to symmetric. Rest days is the exception: almost every value is 0 and Uruguay's 76-day gap
gives a skewness of {num(rest['raw'])} in the training data. FIFA rank and Elo are nearly the same signal
(see the heatmap), and the straight-line vs LOWESS plots show no clear curves; the x² test was significant for
{nonlinear}, so no extra non-linear transformation was needed.

![Correlation](../../images/04_correlation_heatmap.png)

## Preparing the variables (weeks 8 and 9)

We followed the week 9 sequence with statsmodels on the 62 training matches:

| Step | Variables | R² | Adj. R² | AIC | Significant at 5% | Max VIF |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {s['step']} | {s['variables']} | {num(s['R2'], 3)} | {num(s['adj_R2'], 3)} | {num(s['AIC'], 1)} | "
                f"{s['significant_variables']} | {num(s['max_VIF'], 1)} |" for s in steps) + f"""

With all eight raw variables no coefficient is significant, because FIFA rank and Elo carry the same information
(VIF {num(steps[0]['max_VIF'], 1)}). Z-scoring puts the variables on one scale but does not change the fit.
Winsorising and the Yeo-Johnson power transform pulled the rest-days skewness from {num(rest['raw'])} to
{num(rest['winsorised + Yeo-Johnson'])}. Dropping FIFA rank brings every VIF under 5 and raises adjusted R², and
backward elimination ends with the {names(r['backward_elimination_keeps'])} difference alone.

![Transformation](../../images/08_transformation.png)

## Competing models

Every model was cross-validated on the training set (5 folds, repeated {CV_REPEATS} times) and scored once on
the test set. The mean predictor, the Elo-only regression and backward elimination are kept as reference models: they
show whether the full eight-variable model is worth it.

{model_table(r['cv'], best)}

The rule for choosing the final model was **accuracy first, then validity**: start from the candidate with the lowest
CV RMSE, and among the candidates that are not significantly worse (corrected resampled t-test), take the most accurate
one whose residuals pass every assumption test. That gave **{best}**. Compared with the raw eight-variable
regression its CV RMSE fell from {num(raw_cv['cv_RMSE'], 3)} to {num(r['cv'][best]['cv_RMSE'], 3)}
({num(r['cv_improvement_vs_raw_mlr_pct'], 1)}% lower). The Elo-only regression is statistically tied with it
({pval(r['p_selected_vs']['Simple LR: Elo only'])}), which is itself a finding: the other seven variables add little once
Elo is known.

![Model comparison](../../images/09_model_comparison.png)

![Optimisation path](../../images/10_optimisation_path.png)

## The final model on the 42 test matches

| Model | MAE | RMSE | NRMSE | R² |
|---|---:|---:|---:|---:|
""" + "\n".join(f"| {name} | {num(m['MAE'], 3)} | {num(m['RMSE'], 3)} | {num(m['NRMSE'], 3)} | {num(m['R2'], 3)} |"
                for name, m in r["test_metrics"].items()) + f"""

The bootstrap 95% interval for the test R² is {num(ci['R2']['ci_low'], 2)} to {num(ci['R2']['ci_high'], 2)} and for
the RMSE {num(ci['RMSE']['ci_low'])} to {num(ci['RMSE']['ci_high'])}: 42 matches is not many, so these are wide.
Turning the predictions into results, the model picked the correct winner in {r['winners_correct']} of
{r['decisive_test_matches']} decisive matches and got {num(100 * r['result_accuracy'], 0)}% of win/draw/loss outcomes
right. We also simulated every test match 10,000 times (week 11) to get win, draw and loss probabilities; their Brier
score was {num(r['brier_model'], 3)} against {num(r['brier_baseline'], 3)} for a baseline that always predicts the
training-set result shares.

The residuals of the final model pass every check ({assumption_line(r['assumptions'])}). Cook's distance flags
{r['influential_matches']} training matches as influential; we looked at them and kept them, because they are real
results rather than errors. The biggest misses were blow-outs such as {misses[0]['Team_A']} v {misses[0]['Team_B']}
(actual {misses[0]['actual_GD']}, predicted {num(misses[0]['predicted_GD'], 1)}); pre-match information cannot see
a blow-out like that coming.

![Actual vs predicted](../../images/11_actual_vs_predicted.png)

![Outcome probabilities](../../images/19_outcome_probabilities.png)

## Which variables matter

Elo is the most important variable in every view: the Lasso path, the standardised coefficients and permutation
importance. Ridge spreads the weight between Elo and FIFA rank because they are so correlated, while Lasso keeps
the {kept} differences.

![Permutation importance](../../images/16_permutation_importance.png)

## How stable is this?

A single 60/40 split is noisy, so we repeated the split {N_SPLITS} times. Mean test R² was
{num(robust[best]['R2_mean'], 3)} (SD {num(robust[best]['R2_std'], 3)}) for {best},
{num(robust['Simple LR: Elo only']['R2_mean'], 3)} for the Elo-only regression and
{num(robust['MLR: z-score']['R2_mean'], 3)} (SD {num(robust['MLR: z-score']['R2_std'], 3)}) for the raw
eight-variable regression, which falls apart on some splits.

![Robustness](../../images/18_robustness.png)

## Limitations

The test set has only 42 matches, so every test number comes with a wide interval. Goals are rare and random: the
Objective 2.2 report estimates that even a model knowing each team's true scoring rate would have an RMSE of roughly
1.7 for goal difference on 42 matches, which is about where this model sits. Line-ups, injuries and red cards are not in
the data. Team A is usually the stronger side, which the intercept test above accounts for.
"""
    path = folder / "objective_2_1_report.md"
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


def readme_2_1(r):
    best = r["selected_model"]
    mean_gd, team_a, after_elo, favourite, knockout = r["inferential"]
    steps = r["ols_steps"]
    robust = r["robustness"]
    by_test = {a["test"]: a for a in r["assumptions"]}
    dw = next(a["statistic"] for a in r["assumptions"] if a["test"].startswith("Durbin"))
    holds = all(a["p_value"] > ALPHA for a in r["assumptions"] if a["p_value"] == a["p_value"])
    rank_elo = pd.read_csv(DATA_2_1)[["FIFA_Rank_Diff", "Elo_Rating_Diff"]].corr().iloc[0, 1]
    uses = (f"all eight variables in, it keeps the {names(r['lasso_keeps'])} differences" if best.startswith("Lasso")
            else f"uses the {names(r['variables_used'])} differences")
    p_elo = r["p_selected_vs"]["Simple LR: Elo only"]
    kept = r["backward_elimination_keeps"]
    n_sig = steps[0]["significant_variables"]
    significant = "no significant coefficient" if n_sig == 0 else f"only {n_sig} significant coefficient(s)"
    ending = (f"ends with the {names(kept)} difference alone" if len(kept) == 1
              else f"keeps the {names(kept)} differences")
    return f"""## Results – Objective 2.1

**Selected model: {short(best)}** ({uses}).

{test_table(r['test_metrics'], best, '42 test matches')}

- Cross-validated RMSE fell from {num(r['cv']['MLR: z-score']['cv_RMSE'], 3)} for the raw eight-variable regression to
  {num(r['cv'][best]['cv_RMSE'], 3)} ({num(r['cv_improvement_vs_raw_mlr_pct'], 0)}% lower).
- The model picked the winner in **{r['winners_correct']} of {r['decisive_test_matches']}** test matches that had one.
  Its simulated win/draw/loss probabilities have a Brier score of {num(r['brier_model'], 3)}, against
  {num(r['brier_baseline'], 3)} for the baseline.
- {"All residual checks pass" if holds else "Residual checks"}: Shapiro-Wilk {pval(by_test['Shapiro-Wilk']['p_value'])},
  Breusch-Pagan {pval(by_test['Breusch-Pagan']['p_value'])}, RESET {pval(by_test['Ramsey RESET']['p_value'])},
  Durbin-Watson {num(dw)}.
- The raw OLS has {significant}, because FIFA rank and Elo are nearly the same signal (r = {signed(rank_elo, 2)},
  VIF {num(steps[0]['max_VIF'], 0)}). Dropping FIFA rank brings every VIF under {num(steps[3]['max_VIF'], 1)}, and
  backward elimination {ending}.
- **Elo carries most of the signal.** The Elo-only regression is {tied(p_elo, short(best))}. Over {N_SPLITS} random 60/40
  splits its mean test R² is {num(robust['Simple LR: Elo only']['R2_mean'], 3)}, against
  {num(robust[best]['R2_mean'], 3)} for {short(best)}. The raw eight-variable regression averages
  {signed(robust['MLR: z-score']['R2_mean'], 2)} with a standard deviation of {num(robust['MLR: z-score']['R2_std'], 2)},
  which shows why the transformations and penalties matter.
- Team A wins by {num(mean_gd['estimate'])} goals on average (95% CI {num(mean_gd['ci_low'])}–{num(mean_gd['ci_high'])},
  one-sample t-test {pval(team_a['p_value'])}), but once Elo is in the model the advantage
  {"disappears" if after_elo['p_value'] > ALPHA else "remains"} ({pval(after_elo['p_value'])}): Team A is usually
  the stronger team. The higher-Elo team won {num(100 * favourite['estimate'], 0)}% of matches that had a winner
  (95% CI {num(100 * favourite['ci_low'], 0)}–{num(100 * favourite['ci_high'], 0)}%), and knockout games were
  {"not " if knockout['p_value'] > ALPHA else ""}significantly closer ({pval(knockout['p_value'])}).
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
    """The Objective 2.1 results Markdown for a README, from the latest results_summary.json."""
    r1 = load_results()
    return [readme_2_1(r1)] if r1 else []


def update_readme(readme=README):
    """Rewrite the results part of this objective's README with the numbers from the latest full run."""
    return write_readme_results(readme, readme_sections())
