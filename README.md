# FIFA World Cup 2026 – Linear Regression (HIT140 Assessment 3)

Two linear regression models on the 104 matches of the 2026 World Cup, using only pre-match information:

- **Objective 2.1** – predicts a match's goal difference (Team A − Team B). 104 rows. [`objective-goal-difference-2.1/`](objective-goal-difference-2.1/README.md)
- **Objective 2.2** – predicts goals scored by one team. 208 rows. [`objective-goal-count-2.2/`](objective-goal-count-2.2/README.md)

Final report and presentation: [`documents-link.txt`](documents-link.txt)

## Team

| Student | ID | Responsibility |
|---|---|---|
| Hemanta Adhikari | s403355 | Objective 2.1 data, EDA, preprocessing, modelling |
| Birat Adhikari | s403356 | Objective 2.2 data, EDA, preprocessing, modelling |
| Muhammad Hussnain | s406259 | Model experiments, selection, assumption checks |
| John Karki | s403518 | Data loading and checks, evaluation, visualisation |

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                   # both objectives (~4 min)
python main.py --quick           # fast check (<1 min)
python main.py --objective 2.2   # one objective only
```

All random steps use `random_state = 0`, so results are reproducible.

## Data sources

[FIFA](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics) ·
[The Stats Don't Lie](https://www.thestatsdontlie.com/football/world-cup-2026/) ·
[FBref](https://fbref.com/en/)

<!-- results:start - written by python main.py or scripts/run.py, edits here are overwritten -->

## Results

| Objective | Selected model | Test RMSE | Test R² |
|---|---|---:|---:|
| 2.1 Goal difference | Lasso | 1.648 | 0.313 |
| 2.2 Team goals | Ridge + log(1 + goals) | 1.188 | 0.175 |

Elo rating difference is the strongest predictor in both models. Full details are in each objective's README and the final report.

<!-- results:end -->

## Project structure

```
main.py                          runs everything
objective-goal-difference-2.1/   data, model, figures, outputs for 2.1
objective-goal-count-2.2/        data, model, figures, outputs for 2.2
requirements.txt
```