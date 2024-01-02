"""Hyperparameter optimization implementing XGboost and RandomForest. Models are trainned in a sequential fashion using the TimeSeriesSplit for a k-fold validation for TimeSeries. Hyperparmeter selection for RandomForest and XGBoost using GridSearchCV to iterate over a heuristic selection of hyperparmeters."""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from forecast.utils.utils import (
    current_date_formated,
    perform_standardization,
    preprocess_data,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

# TODO: Can't user import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

# TODO: Ruff is forcing us to define the following instructions after the import. For testing this script separately rearange the order of these lines of code.
current_dir = os.path.dirname(os.path.realpath(__file__))
# Append the 'src' directory to the Python path
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(src_dir)
logging.basicConfig(level=logging.INFO)


def split_data(
    df: pd.DataFrame, date_col: str, test_cut: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide the the inital DatSet into train and test to compare the results after the model has been trained and the predictions are stores.

    Args:
        df (pd.DataFrame): DataFrame with all the information.
        date_col (str): Date column name.
        test_cut (datetime): Date that divides train from test

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: _description_
    """
    training_mask = df[date_col] < test_cut
    training_data = df.loc[training_mask]

    testing_mask = df[date_col] >= test_cut
    testing_data = df.loc[testing_mask]
    return training_data, testing_data


def evaluate_model(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    """Prints Mean Squared Error and Mean Absolute Error.

    Args:
        y_true (pd.DataFrame): Real values for commits per week.
        y_pred (pd.DataFrame): Predicted values.
    """
    _y_true = y_true[~y_true.isna()]
    _y_pred = y_pred[~y_true.isna()]

    mse = mean_squared_error(_y_true, _y_pred)
    mape = mean_absolute_percentage_error(_y_true, _y_pred)
    logging.info(f"MSE: {mse}")
    logging.info(f"MAPE: {mape}")


def hyperparameter_optimization(
    df: pd.DataFrame,
    prediction_window,
    evaluation_window,
    cut_date,
    target="commit_count",
    subfix: str = None,
):
    """Iterates over the train set to make sequential predictions. Train models using a set of hyperparamters and returns the best combination of hyperparamters trough the scikit larn GridSearch library. Train data for each iteration is selected using the TimeSeries split that sequentially selects data for the next prediction.

    Args:
        df (pd.DataFrame): DataFrame passed by the feature engineering process.
        subfix (str, optional): Optional label to save data. Defaults to None.
        target (str, optional): Predictions are created over this variable. Defaults to "commit_count".
    """
    # Splitting Data
    scalers_dict = {}
    for col in df.columns:
        if col not in ["repo_name", "date", "year", "month", "week"]:
            df, scaler = perform_standardization(df, col, method="z-score")
            scalers_dict[col] = scaler
    test_cut = (
        cut_date - timedelta(days=prediction_window) - timedelta(days=evaluation_window)
    )
    # Notice that we are going to be placed at t='2021-12-26'  the we need to start the iteration at MAX date to cover predictions until '2021-12-26'. Notice also that you will need at least EVALUATION_WINDOW observations before max date in order to create predictions for MAX_DATE. Nex. we just need to advance one day at a time to recalculate the predictions using previous predictions.
    training_data, testing_data = split_data(df, "date", test_cut)
    # Prepare training and testing sets for XGBoost model
    # DELETE ALL UNAVAILABLE COLUMNS at time t
    features = [
        x
        for x in df.columns
        if x
        not in ["repo_name", "week", "date", "year", "month", "index", "commit_count"]
        # TODO: include seasonal variables as Dummies
    ]

    y_test = testing_data[target]
    X_test = testing_data[features]
    X_train = training_data[features]
    y_train = training_data[target]

    # XGBOOST
    model_filename = f"models/{target}_xgboost.joblib"
    # Save the best model to a file
    grid_search_xgboost = train_xgboost_model(X_train, y_train, prediction_window)
    best_estimator = grid_search_xgboost.best_estimator_
    best_params = grid_search_xgboost.best_params_
    # Print the best hyperparameters
    logging.info("\tbest_params", best_params)
    joblib.dump(best_estimator, model_filename)

    prediction_xgboost = grid_search_xgboost.predict(X_test)
    evaluate_model(y_test, prediction_xgboost)
    save_results(
        prediction_xgboost,
        testing_data,
        model_filename,
        model="xgboost",
        subfix=subfix,
    )

    if False:  # TODO: activate when deploying on EC2
        model_filename = f"./models/{target}_lgbm.joblib"

        grid_search_lgbm = train_lgbm_model(X_train, y_train)
        best_estimator = grid_search_lgbm.best_estimator_
        best_params = grid_search_lgbm.best_params_

        # Save the best model to a file
        joblib.dump(best_estimator, model_filename)
        prediction_lgbm = grid_search_lgbm.predict(X_test)
        evaluate_model(y_test, prediction_lgbm)
        save_results(
            prediction_lgbm, testing_data, model_filename, model="lgbm", subfix=subfix
        )

    # RANDOM_FOREST:
    model_filename = f"./models/{target}_randomforest.joblib"
    grid_search_rf = train_randomforest_model(X_train, y_train)
    best_estimator = grid_search_rf.best_estimator_
    best_params = grid_search_rf.best_params_

    # Save the best model to a file
    joblib.dump(best_estimator, model_filename)
    prediction_randomf = grid_search_rf.predict(X_test)
    evaluate_model(y_test, prediction_randomf)
    save_results(
        prediction_randomf,
        testing_data,
        model_filename,
        model="randomforest",
        subfix=subfix,
    )


def save_results(
    prediction: pd.Series,
    testing_data: pd.DataFrame,
    model_filename: str,
    model: str,
    subfix: str = None,
):
    """Save predicted values over the trainning set, this information is not used in the following steps but it can be used to evaluate each model independently.

    Args:
        prediction (pd.Series): _description_
        testing_data (pd.DataFrame): _description_
        model_filename (str): _description_
        model (str): _description_
        subfix (str, optional): _description_. Defaults to None.
    """
    y_test = testing_data["commit_count"]
    df = pd.DataFrame()
    df["Prediction"] = prediction
    df[date_column] = testing_data[date_column].values
    df["Model"] = model_filename
    df["Actual"] = y_test.values
    df["Target"] = "commit_count"

    df_M = df.copy()
    df_M["Year"] = df_M.date.dt.year
    df_M["Month"] = df_M.date.dt.month
    df_M[date_column] = pd.to_datetime(
        df_M["Year"].astype(str) + "-" + df_M["Month"].astype(str) + "-01"
    )
    df_M = df_M.groupby(date_column).sum().reset_index()
    df_M["Error"] = df_M["Actual"] - df_M["Prediction"]
    df_M["pct_abs_error"] = np.abs(df_M["Error"])
    mape = np.mean(df_M["pct_abs_error"])
    df_M["MAPE"] = mape

    if SAVE_MODEL_PERFORMANCE:
        save_path = f"../data/process/{model}_commit_count_{subfix}"
        df.to_csv(save_path + "_W.csv", index=False)
        df_M.to_csv(save_path + "_M.csv", index=False)


def train_xgboost_model(X_train, y_train, n_splits=4, test_size=100):
    """Train an XGBoost regression model using GridSearchCV.

    Parameters:
    X_train (DataFrame): Training feature data.
    y_train (Series): Training target data.

    Returns:
    grid_search (GridSearchCV): Trained model with the best hyperparameters.
    """
    model = XGBRegressor()
    parameters = {
        "max_depth": [6, 9],
        "learning_rate": [0.01, 0.03],
        "n_estimators": [80, 100, 150],
        "colsample_bytree": [0.2, 0.3],
    }
    cv_split = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    grid_search = GridSearchCV(estimator=model, cv=cv_split, param_grid=parameters)
    # TODO: create iterative prediction recreating lag and roll variables.
    grid_search.fit(X_train, y_train)

    return grid_search


def train_lgbm_model(X_train, y_train, n_splits=4, test_size=100):
    """Train a LightGBM regression model using GridSearchCV.

    Parameters:
    X_train (DataFrame): Training feature data.
    y_train (Series): Training target data.

    Returns:
    grid_search (GridSearchCV): Trained model with the best hyperparameters.
    """
    model = "Not supported"  # lgb.LGBMRegressor()
    parameters = {
        "max_depth": [6, 10],
        "num_leaves": [20, 25],
        "learning_rate": [0.017, 0.05],
        "n_estimators": [50, 70],
        "colsample_bytree": [
            0.3,
            0.5,
        ],
    }
    cv_split = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    grid_search = GridSearchCV(
        estimator=model,
        cv=cv_split,
        verbose=0,
        param_grid=parameters,
        scoring=make_scorer(
            mean_absolute_error, greater_is_better=False
        ),  # Specify MAE scorer
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def train_randomforest_model(X_train, y_train, n_splits=4, test_size=100):
    """Train a Random Forest regression model using GridSearchCV.

    Parameters:
    X_train (DataFrame): Training feature data.
    y_train (Series): Training target data.

    Returns:
    grid_search (GridSearchCV): Trained model with the best hyperparameters.
    """
    model = RandomForestRegressor()
    parameters = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt"],
    }
    cv_split = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    grid_search = GridSearchCV(
        estimator=model,
        cv=cv_split,
        verbose=0,
        param_grid=parameters,
        scoring=make_scorer(
            mean_absolute_error, greater_is_better=False
        ),  # Specify MAE scorer
    )
    grid_search.fit(X_train, y_train)
    return grid_search


# Plotting Functions
def plot_actual_vs_predicted_to_file(
    testing_dates,
    y_test,
    prediction,
    level: str,
    save_path: str,
    target: str = "commit_count",
):
    """Create a plot that compares predicted data points returned by the ML models and compares it with the real values.

    Args:
        df (pd.DataFrame): _description_
        prediction (str): Prediction obtained by ML model.
        y_test (str): Real values, commits reported per weel.
        title (str): title for the plot.
        save_path (str): File name to save plot as png.
        level (str): Report level of the plot.
        target (str, optional): Name of the target variable. Defaults to "commit_count".

    """
    if level == "default":
        df_test = pd.DataFrame(
            {date_column: testing_dates, "actual": y_test, "prediction": prediction}
        )
    elif level == "monthly":
        df_test = pd.DataFrame(
            {date_column: testing_dates, "actual": y_test, "prediction": prediction}
        )
        df_test["MONTH"] = df_test[date_column].dt.month.apply(lambda x: int(x))
        df_test["YEAR"] = df_test[date_column].dt.year.apply(lambda x: int(x))
        df_test[date_column] = pd.to_datetime(
            df_test["YEAR"].astype(str) + "-" + df_test["MONTH"].astype(str) + "-01"
        )
        df_test = df_test.groupby([date_column]).sum()
        df_test = df_test.reset_index()

    plt.figure(figsize=(16, 10))  # Set figure size
    plt.plot(df_test[date_column], df_test["actual"], label="Actual")
    plt.plot(df_test[date_column], df_test["prediction"], label="Prediction")
    plt.xlabel(date_column)
    plt.ylabel(target)
    plt.title("Actual vs. Predicted Quantity")
    plt.legend()

    # Save the plot as a PNG image
    plt.savefig(
        save_path, dpi=100, bbox_inches="tight"
    )  # Save with 100 dpi and tight layout
    plt.close()  # Close the plot to release memory


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    TEST = True
    SAVE_MODEL_PERFORMANCE = True
    # From the previous script
    lag_list = [2, 4, 6, 10]
    rolling_list = [2, 4, 6]
    date_column = "date"
    subfix = current_date_formated()
    evaluation_window = (
        max(lag_list) * 7 + max(rolling_list) * 7
    )  # minimum data to run t
    prediction_window = 7 * 12  # days_in_week*number_weeks
    cut_date = pd.to_datetime("2021-12-26")

    if TEST:
        file_path = "../data/preprocess/featureengineering_test.csv"
    else:
        file_path = "../data/preprocess/featureengineering.csv"

    df = pd.read_csv(file_path)
    df = df[~df["commit_count"].isna()]

    # Data Preprocessing
    df = preprocess_data(df, date_column)  # 30 days
    hyperparameter_optimization(
        df,
        "",
        "commit_count",
        prediction_window=prediction_window,
        evaluation_window=evaluation_window,
        cut_date=cut_date,
    )
