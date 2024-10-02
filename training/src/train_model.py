import warnings

warnings.filterwarnings(action="ignore")

from functools import partial
from typing import Callable

import hydra
import joblib
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def load_data(path: DictConfig):
    X_train = pd.read_csv(abspath(path.X_train.path))
    X_test = pd.read_csv(abspath(path.X_test.path))
    y_train = pd.read_csv(abspath(path.y_train.path))
    y_test = pd.read_csv(abspath(path.y_test.path))
    return X_train, X_test, y_train, y_test


def get_objective(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    config: DictConfig,
    space: dict,
):

    model = RandomForestClassifier(
        n_estimators=int(space["n_estimators"]),
        max_depth=int(space["max_depth"]),
        min_samples_split=int(space["min_samples_split"]),
        min_samples_leaf=int(space["min_samples_leaf"]),
        max_features=space["max_features"],
        bootstrap=space["bootstrap"],
        random_state=config.model.seed,
    )

    model.fit(X_train, y_train.values.ravel())
    prediction = model.predict(X_test.values)
    accuracy = accuracy_score(y_test, prediction)
    print("SCORE:", accuracy)
    return {"loss": -accuracy, "status": STATUS_OK, "model": model}


def optimize(objective: Callable, space: dict):
    trials = Trials()
    best_hyperparams = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
    )
    print("The best hyperparameters are : ", "\n")
    print(best_hyperparams)
    best_model = trials.results[
        np.argmin([r["loss"] for r in trials.results])
    ]["model"]
    return best_model


@hydra.main(config_path="../../config", config_name="main")
def train(config: DictConfig):
    """Function to train the model"""
    print(f"Using configuration file: {abspath('main.yaml')}")

    X_train, X_test, y_train, y_test = load_data(config.processed)

    # Define space
    space = {
        "max_depth": hp.quniform("max_depth", **config.model.max_depth),
        "min_samples_split": hp.quniform("min_samples_split", **config.model.min_samples_split),
        "min_samples_leaf": hp.quniform("min_samples_leaf", **config.model.min_samples_leaf),
        "max_features": hp.uniform("max_features", config.model.max_features.low, config.model.max_features.high),
        "bootstrap": hp.choice("bootstrap", [config.model.bootstrap]),
        "n_estimators": config.model.n_estimators,
    }
    objective = partial(
        get_objective, X_train, y_train, X_test, y_test, config
    )

    # Find best model
    best_model = optimize(objective, space)

    # Save model
    joblib.dump(best_model, abspath(config.model.path))


if __name__ == "__main__":
    train()