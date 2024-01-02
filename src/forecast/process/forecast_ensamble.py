"""Ensamble model that used pretrained models and combine them with an elasticnet regression. Additional seasonal and cluster variables are generated."""

import logging
import os
import sys
from datetime import timedelta
from types import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans

current_dir = os.path.dirname(os.path.realpath(__file__))
# Append the 'src' directory to the Python path
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(src_dir)
logging.basicConfig(level=logging.INFO)

# TODO: Ruff is forcing us to define the following instructions after the import. For testing this script separately rearange the order of these lines of code.
from forecast.utils.utils import convert_to_first_monday_of_week


def prepare_predictions(path_original: str, retrain_models: bool) -> pd.DataFrame:
    """Prepare predictions DataFrame based on the original data.

    Args:
        path_original (str): The file path of the original data.
        retrain_models (bool): Flag indicating whether to retrain models or not.

    Returns:
        pd.DataFrame: DataFrame containing the prepared predictions with columns:
            - Date: Date of the prediction.
            - model_family: Model family associated with the predictions.
            - Commit Real: Actual commit counts.
            - Commit Forecast: Forecasted commit counts.
            - Repository: Repository name.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
    """
    df_forecast_D = pd.DataFrame()

    try:
        df = pd.read_csv(path_original)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found at path: {path_original}") from e

    model_path_parts = path_original.split("/")[-1].split("_")
    model = model_path_parts[-2]
    df["model_family"] = model
    df_forecast_D = df.copy()
    df_forecast_D["Date"] = pd.to_datetime(df_forecast_D.Date)
    df_forecast_commits = df_forecast_D[
        ["Date", "model_family", "Commit Real", "Commit Forecast", "Repository"]
    ]
    df_forecast_commits = df_forecast_commits[df_forecast_commits.Date < cut_date]

    if not retrain_models:  # We don't have the actual values.
        df_forecast_commits["Commit Forecast"] = np.nan

    return df_forecast_commits


def prepare_all_models(retrain_models: bool) -> pd.DataFrame:
    """Prepare predictions for all models.

    Args:
        retrain_models (bool): Flag indicating whether to retrain models or not.

    Returns:
        pd.DataFrame: DataFrame containing the prepared predictions for all models with columns:
            - Date: Date of the prediction.
            - model_family: Model family associated with the predictions.
            - Commit Real: Actual commit counts.
            - Commit Forecast: Forecasted commit counts.
            - Repository: Repository name.

    Raises:
        FileNotFoundError: If any of the model prediction files is not found.
    """
    df_forecast_prepared_all = pd.DataFrame()

    for model in tqdm(["randomforest", "xgboost"]):
        path_original = f"../data/process/predictions_{model}.csv"
        df_forecast_prepared = prepare_predictions(path_original, retrain_models)
        if len(df_forecast_prepared_all) == 0:
            df_forecast_prepared_all = df_forecast_prepared
        else:
            df_forecast_prepared_all = pd.concat(
                [df_forecast_prepared, df_forecast_prepared_all], ignore_index=True
            )

    return df_forecast_prepared_all


def test_uniqueness_branch_date(df: pd.DataFrame) -> None:
    """Test the uniqueness of branch and date combinations in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing commit data with columns:
            - Repository: Repository name.
            - Date: Date of the commit.

    Raises:
        ValueError: If there are duplicate entries for any repository and date combination.
    """
    for repo in df.Repository.unique():
        df_b = df[df.Repository == repo]
        for date in df_b.Date.unique():
            df_db = df_b[df_b.Date == date]
            if len(df_db) > 1:
                raise ValueError(
                    "Duplicate entries found for repository and date combination."
                )


def weekly_group(
    df: pd.DataFrame,
    date_col: str,
    index_cols: List[str],
    sum_cols: List[str],
    mean_cols: List[str],
) -> pd.DataFrame:
    """Group the DataFrame at a weekly level based on specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Name of the date column.
        index_cols (List[str]): List of columns to use as index for grouping.
        sum_cols (List[str]): List of columns to sum during grouping.
        mean_cols (List[str]): List of columns to calculate mean during grouping.

    Returns:
        pd.DataFrame: Grouped DataFrame.

    Raises:
        ValueError: If the difference between the sum/mean before and after grouping exceeds a threshold.
    """
    if len(sum_cols) > 0:
        test_col = sum_cols[0]
        total_test_1 = df[test_col].sum()
    else:
        test_col = mean_cols[0]
        total_test_1 = df[test_col].mean()

    merge_group = index_cols.copy()
    merge_group.append(date_col)
    df[date_col] = pd.to_datetime(df[date_col])
    df[date_col] = df[date_col].apply(lambda x: convert_to_first_monday_of_week(x))
    df_week_sum = df.groupby(merge_group)[sum_cols].sum().reset_index()
    df_week_mean = df.groupby(merge_group)[mean_cols].mean().reset_index()

    df_out = df_week_mean.merge(df_week_sum, on=merge_group, how="inner")

    if len(sum_cols) > 0:
        total_test_2 = df_out[test_col].sum()
        if total_test_1 - total_test_2 > 0.01:
            raise ValueError("Difference between sum values exceeds the threshold.")
    else:
        total_test_2 = df_out[test_col].mean()
        if np.abs(total_test_1 - total_test_2) / total_test_1 > 0.2:
            raise ValueError(
                "Relative difference between mean values exceeds the threshold."
            )

    df_out[date_col] = pd.to_datetime(df_out[date_col])
    df_out = df_out.sort_values(index_cols + [date_col], ascending=True)
    return df_out


def train_elastic_net_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 3,
    test_size: int = 100,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
) -> ElasticNet:
    """Train an Elastic Net regression model using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        n_splits (int): Number of splits for time series cross-validation.
        test_size (int): Size of the test set in each split.
        alpha (float): Regularization strength for Elastic Net.
        l1_ratio (float): L1 ratio for combining L1 and L2 penalties.

    Returns:
        ElasticNet: Trained Elastic Net model.
    """
    # Create an instance of the ElasticNet model
    elastic_net_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    parameters = {
        "alpha": [alpha],
        "l1_ratio": [l1_ratio],
    }

    cv_split = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    # Use GridSearchCV with the ElasticNet model
    grid_search = GridSearchCV(
        estimator=elastic_net_model,
        cv=cv_split,
        param_grid=parameters,
        scoring="neg_mean_squared_error",
    )
    grid_search.fit(X_train, y_train)

    return grid_search


def create_forecast_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a dummy variable 'horizon_indicator' based on the forecast horizon.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'Date' column.

    Returns:
        pd.DataFrame: DataFrame with the added 'horizon_indicator' column.
    """
    # Fit and transform the 'Cluster_Labels' column to get dummy variables
    df["Date"] = pd.to_datetime(df.Date)
    df["horizon_indicator"] = 1000
    # TODO: This is prone to errors, is better to pass a GLOBAL var that tells the starting month of the forecast
    min_forecast_date = df.Date.min()
    fhorizon_cont = 1
    for i in range(min_forecast_date.month, 13):
        forecast_month_mask = df["Date"].dt.month == i
        df.loc[forecast_month_mask, "horizon_indicator"] = fhorizon_cont
        fhorizon_cont += 1
    for i in range(1, min_forecast_date.month):
        forecast_month_mask = df["Date"].dt.month == i
        df.loc[forecast_month_mask, "horizon_indicator"] = fhorizon_cont
        fhorizon_cont += 1
    selected_columns = [
        col
        for col in df.columns
        if col
        not in [
            "Commit Real",
            "Commit Forecast",
            "month",
            "model_family",
            "randomforest",
            "xgboost",
        ]
    ]  # TODO: Add "lgbm"
    return df[selected_columns]


def create_month_dummy(df: pd.DataFrame, retrain_models: bool):
    """Creates dummy variables for each month of the year based on the 'Date' column.

    This function is designed to be part of a script that trains an ensemble model using
    previous predictions from machine learning models and generates a forecast over the
    commits created in Git for each repository at a weekly level. The dummy variables
    represent the month in which the commits occurred, providing a categorical feature
    for modeling monthly patterns.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'Date' column.
        retrain_models (bool): Flag indicating whether to retrain the dummy encoder.

    Returns:
        pd.DataFrame: DataFrame with added dummy variables for each month.
    """
    # Fit and transform the 'Cluster_Labels' column to get dummy variables
    df["Date"] = pd.to_datetime(df.Date)
    df["month"] = df.Date.dt.month
    encoder_filename = os.path.abspath("./models/dummy_encoder.joblib")

    if retrain_models:
        # Train and save the dummy encoder
        encoder = OneHotEncoder(sparse=False)
        encoder = encoder.fit(df[["month"]])
        joblib.dump(encoder, encoder_filename)
    else:
        # Load the pre-trained dummy encoder
        encoder = joblib.load(encoder_filename)

    dummy_variables = encoder.transform(df[["month"]])
    # Create a new DataFrame with dummy variable column names
    df_dummy = pd.DataFrame(
        dummy_variables,
        columns=[f"Month_{i+1}" for i in range(dummy_variables.shape[1])],
    )

    # Append dummy variable columns to the original DataFrame
    df = pd.concat([df, df_dummy], axis=1)
    # Remove unnecessary columns
    selected_columns = [
        col
        for col in df.columns
        if col
        not in [
            "model_family",
            "Commit Real",
            "Commit Forecast",
            "month",
            "xgboost",
            "randomforest",
        ]
    ]  # TODO: add 'lgbm',
    return df[selected_columns]


def create_cluster(
    df: pd.DataFrame,
    n: int = 8,
    target: str = "commit_count",
    retrain_models: bool = True,
) -> np.ndarray:
    """Creates clusters based on the 'target' last year trends, assigning each branch to a different cluster.

    This function is designed to analyze time series behavior for each branch in a repository
    and create clusters based on the trends observed in the specified 'target' column (e.g., 'commit_count').
    The clustering is performed using Time Series K-Means, and the resulting cluster labels are returned.

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data.
        n (int, optional): Number of clusters to create. Defaults to 8.
        target (str, optional): Name of the target column for clustering. Defaults to "commit_count".
        retrain_models (bool, optional): Flag indicating whether to retrain the clustering model. Defaults to True.

    Returns:
        np.ndarray: Array of cluster labels assigned to each branch.

    Note:
        Ensure that the 'date' and 'repo_name' columns are present in the DataFrame for proper clustering.

    Example:
        ```
        df_clustered = create_cluster(df, n=8, target="commit_count", retrain_models=True)
        ```
    """
    # Extract time series data for each branch
    time_series_data = []
    branch_ids = df["repo_name"].unique()
    df_week_series = weekly_group(
        df,
        date_col="date",
        index_cols=["repo_name"],
        sum_cols=[target],
        mean_cols=[],
    )

    for branch_id in branch_ids:
        branch_data = df_week_series[df_week_series["repo_name"] == branch_id][target]
        time_series_data.append(branch_data.values)

    # Standardize time series data
    scaler = StandardScaler()
    time_series_data_fitted = [
        scaler.fit_transform(ts.reshape(-1, 1)) for ts in time_series_data
    ]

    if retrain_models:
        # Perform Time Series K-Means clustering
        cluster_filename = os.path.abspath("./models/cluster.joblib")
        model = TimeSeriesKMeans(n_clusters=n, verbose=True, random_state=0)
        model = model.fit(time_series_data_fitted)
        predicted_labels = model.predict(time_series_data_fitted)
        joblib.dump(model, cluster_filename)
    else:
        cluster_filename = os.path.abspath(
            "./models/cluster.joblib"
        )  # Last time model was tuned with 1 year data
        model = joblib.load(cluster_filename)
        predicted_labels = model.predict(time_series_data_fitted)

    return predicted_labels


def cluster_indicator(df: pd.DataFrame, predicted_labels: np.ndarray) -> pd.DataFrame:
    """Creates a dummy variable for each of the clusters.

    Each branch gets assigned 1 for only 1 of the clusters, and then it reports a value of 0 for all
    the other cluster control variables.

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data.
        predicted_labels (np.ndarray): Array of cluster labels assigned to each branch.

    Returns:
        pd.DataFrame: DataFrame with cluster indicators as dummy variables.

    Note:
        Ensure that the 'repo_name' and 'cluster' columns are present in the DataFrame.

    Example:
        ```
        df_clustered = create_cluster(df, n=8, target="commit_count", retrain_models=True)
        df_cluster_indicator = cluster_indicator(df, df_clustered)
        ```
    """
    # Add the cluster labels to the original DataFrame
    cluster_labels = []
    branch_ids = df["repo_name"].unique()

    for label, branch_id in zip(predicted_labels, branch_ids):
        cluster_labels.extend([label] * len(df[df["repo_name"] == branch_id]))

    df["cluster"] = cluster_labels
    df = (
        df.groupby(["repo_name", "cluster"])
        .first()
        .reset_index()[["repo_name", "cluster"]]
    )
    df = df.rename(columns={"repo_name": "Repository"})

    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    # Fit and transform the 'Cluster_Labels' column to get dummy variables
    dummy_variables = encoder.fit_transform(df[["cluster"]])
    # Create a new DataFrame with dummy variable column names
    df_dummt = pd.DataFrame(
        dummy_variables,
        columns=[f"Cluster_{i}" for i in range(dummy_variables.shape[1])],
    )
    df = pd.concat([df, df_dummt], axis=1)

    return df


def train_elasticnet_ensemble_model(
    df: pd.DataFrame,
    target: str = "commit_count",
    retrain_models: bool = True,
    load_model: bool = False,
) -> Tuple[ElasticNet, pd.DataFrame]:
    """Train an ensemble model using ElasticNet regression.

    This function prepares the data, trains an ElasticNet model, and makes predictions on the testing data.

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data.
        target (str): Target variable for regression (default is "commit_count").
        retrain_models (bool): Flag to retrain the ElasticNet model (default is True).
        load_model (bool): Flag to load a pre-trained model instead of training (default is False).

    Returns:
        Tuple[ElasticNet, pd.DataFrame]: Trained ElasticNet model and DataFrame with predictions.

    Example:
        ```
        df_ensemble_train = prepare_all_models(retrain_models=True)
        model_elasticnet, df_with_predictions = train_elasticnet_ensemble_model(df_ensemble_train)
        ```

    Note:
        Ensure that the required columns ('commit_count', 'date', 'repo_name', 'index') are present in the DataFrame.
    """
    # Ensure uniqueness of branches and dates
    test_uniqueness_branch_date(df)
    # Prepare training and testing sets for the ElasticNet model
    # Delete all unavailable columns at time t
    features = [
        x
        for x in df.columns
        if x != "commit_count" and x != "date" and x != "repo_name" and x != "index"
    ]

    y = df[target]
    X = df[features]
    del X["Date"]
    del X["Repository"]

    # Create an instance of the ElasticNet model
    model_filename = "./models/modelperformance_elasticnet.joblib"

    if retrain_models:
        grid_search_elasticnet = train_elastic_net_model(
            X, y, n_splits=3, test_size=30, alpha=1.0, l1_ratio=0.5
        )

        # Save the best model to a file
        model = grid_search_elasticnet.best_estimator_
        joblib.dump(model, model_filename)
    else:
        # Load a pre-trained model
        model = joblib.load(model_filename)

    # Make predictions on the testing data
    y_pred = model.predict(X)
    df["elasticnet"] = y_pred

    return model, df


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    date_column = "date"
    TEST = True
    prediction_start = pd.to_datetime("2022-09-01")
    retrain_models = True
    cut_date = pd.to_datetime("2021-12-26")
    prediction_contained = True
    prediction_window = 12
    lag_list = [2, 4, 6, 10]
    rolling_list = [2, 4, 6]
    evaluation_window = max(lag_list) + max(rolling_list) + 1

    if prediction_contained:
        prediction_start = cut_date - timedelta(days=(prediction_window * 7))
        min_date = prediction_start - timedelta(days=(evaluation_window * 7))
        prediction_end = cut_date

    else:
        # Start from the last day and starts making iterations over the future
        prediction_start = cut_date
        min_date = prediction_start - timedelta(days=(evaluation_window * 7))
        prediction_end = cut_date + timedelta(
            days=(prediction_window * 7)
        )  # Prediction of the future

    if TEST:
        all_entries_path = "../data/preprocess/featureengineering_test.csv"
    else:
        all_entries_path = "../data/preprocess/featureengineering.csv"

    df_all = pd.read_csv(all_entries_path)
    df_forecast = prepare_all_models(retrain_models)

    df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])
    df_forecast = df_forecast[
        df_forecast["Commit Forecast"] != df_forecast["Commit Real"]
    ]

    df_forecast = (
        df_forecast[
            ["Repository", "Date", "Commit Forecast", "Commit Real", "model_family"]
        ]
        .groupby(["Date", "Repository", "model_family"])
        .first()
        .reset_index()
    )

    df_models = df_forecast.pivot(
        index=["Repository", "Date"], columns="model_family", values="Commit Forecast"
    ).reset_index()
    df_forecast_month = create_month_dummy(df_models, retrain_models=retrain_models)
    test_uniqueness_branch_date(df_models)
    df_forecast_horizon = create_forecast_horizon(df_models.copy())
    df_target = df_forecast[["Repository", "Date", "Commit Real"]].copy()
    df_target = (
        df_target.groupby(["Repository", "Date"])["Commit Real"].first().reset_index()
    )

    predicted_labels = create_cluster(
        df=df_all, n=8, retrain_models=retrain_models, target="commit_count"
    )
    df_clusters = cluster_indicator(df_all, predicted_labels)

    df_lr = df_models.merge(df_clusters, on="Repository", how="left")
    test_uniqueness_branch_date(df_lr)
    test_uniqueness_branch_date(df_forecast_month)

    df_model_input = df_lr.merge(
        df_forecast_month, on=["Repository", "Date"], how="inner"
    )
    df_model_input = df_model_input.merge(
        df_forecast_horizon, on=["Repository", "Date"], how="inner"
    )
    df_model_input = df_target.merge(
        df_model_input, on=["Repository", "Date"], how="inner"
    )
    test_uniqueness_branch_date(df_model_input)

    model, df_results = train_elasticnet_ensemble_model(
        df_model_input,
        target="Commit Real",
        retrain_models=retrain_models,
        load_model=True,
    )

    df_results["model_family"] = "elasticnet"
    date_prediction_cut = pd.to_datetime("2023-08-01")
    df_forecast = pd.concat(
        [df_forecast, df_results], ignore_index=True
    )  # Check columns

    df_forecast.to_csv("final_predictions.csv", index=False)
