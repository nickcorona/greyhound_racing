import lightgbm as lgb
from helpers import similarity_encode
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"data\forestfires.csv",
    parse_dates=[],
    index_col=[],
)
X, y = similarity_encode(df, encode=False, categorize=True, preran=False)
X = X.drop("rain", axis=1)
d = lgb.Dataset(X, y, silent=True)

# rmse: 98.18188205858038
NUM_BOOST_ROUND = 455
params = {
    "objective": "rmse",
    "metric": "rmse",
    "verbose": -1,
    "n_jobs": 6,
    "learning_rate": 0.004090619790710353,
    "feature_pre_filter": False,
    "lambda_l1": 6.99239231800302e-08,
    "lambda_l2": 9.330959145992983,
    "num_leaves": 9,
    "feature_fraction": 0.8999999999999999,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "min_child_samples": 20,
}

model = lgb.train(params, d, num_boost_round=NUM_BOOST_ROUND)

Path("figures").mkdir(exist_ok=True)
lgb.plot_importance(model, grid=False)
plt.savefig("figures/feature_importange.png")

Path("models").mkdir(exist_ok=True)
model.save_model(
    "models/model.pkl",
    num_iteration=NUM_BOOST_ROUND,
    importance_type="gain",
)
