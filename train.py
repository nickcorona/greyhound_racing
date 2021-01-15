import lightgbm as lgb
from helpers import similarity_encode
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"data\data_final.csv",
    parse_dates=[],
    index_col=[],
)
TARGET = "Winner"
DROPPED_FEATURES = [
    "Odds",
    "Odds_Recent",
    "Races_380",
    "Public_Estimate",
    "Early_Recent",
]
y = df[TARGET]
X = df.drop([TARGET, "Finished", *DROPPED_FEATURES], axis=1)
d = lgb.Dataset(X, y, silent=True)

NUM_BOOST_ROUND = 439
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbose": -1,
    "n_jobs": 6,
    "learning_rate": 0.008591072626027468,
    "feature_pre_filter": False,
    "lambda_l1": 5.13269456420478,
    "lambda_l2": 2.0036887398035568e-08,
    "num_leaves": 2,
    "feature_fraction": 0.92,
    "bagging_fraction": 0.8868019137735326,
    "bagging_freq": 1,
    "min_child_samples": 20,
}

model = lgb.train(params, d, num_boost_round=NUM_BOOST_ROUND)

Path("figures").mkdir(exist_ok=True)
lgb.plot_importance(model, grid=False, figsize=(10, 5))
plt.savefig("figures/feature_importange.png")

Path("models").mkdir(exist_ok=True)
model.save_model(
    "models/model.pkl",
    num_iteration=NUM_BOOST_ROUND,
    importance_type="gain",
)
