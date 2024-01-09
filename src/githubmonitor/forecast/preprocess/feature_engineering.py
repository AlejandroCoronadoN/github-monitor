"""Calculates averages over differnet peridos of time in time series data using different offsets (lags)."""
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

# TODO: Ruff is forcing us to define the following instructions after the import. For testing this script separately rearange the order of these lines of code.
current_dir = os.path.dirname(os.path.realpath(__file__))
# Append the 'src' directory to the Python path
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(src_dir)
logging.basicConfig(level=logging.INFO)

from forecast.utils.utils import (
    generate_date_range,
    get_testing_params,
    interact_categorical_numerical,
)


def featureengineering_test(
    df_out: pd.DataFrame,
    col: str,
    feature: str,
    date_column: str = "date",
    agg_level: str = "W",
):
    """Test if the total amount of commits on the widow average functions matches the means if we take the same observations and calculate the mean using a loop function.

    Args:
        df_out (pd.DataFrame): Output DataFrame to be testes by this function.
        col (str): Average window function to test.
        feature (str, optional): Orginal variable that was used to create the average window variable (col)
        date_column (str): Date column name.

    Raises:
        ValueError: Average window variable should be the same to the value created by this test.
        ValueError: There shouldn't exist na values (after filtering with discard_uncompleted_windows)
    """
    roll, lag = get_testing_params(col)
    cont_na = 0
    for repo in tqdm(df_out.repo_name.unique()):
        df_b = df_out[df_out.repo_name == repo].reset_index()
        df_b = df_b.sort_values(["repo_name", date_column], ascending=True)
        all_dates = df_b[date_column].unique()
        min_roll_date = all_dates.min()
        for date in sorted(all_dates):
            if date > min_roll_date + timedelta(days=roll):
                lag_date = pd.to_datetime(
                    df_b.loc[df_b[date_column] == date, date_column].values[0]
                )

                if agg_level == "W":
                    discarted_observation = timedelta(days=roll * 7)
                elif agg_level == "D":
                    discarted_observation = timedelta(days=roll)
                elif agg_level == "M":
                    discarted_observation = timedelta(days=roll * 30)
                else:
                    raise ValueError("Select a valid agg_level: D, M, W")

                start_date = lag_date - discarted_observation
                test_dates = generate_date_range(start_date, roll)
                mean_dates_mask = df_out[date_column].isin(test_dates)
                repo = "Azure/azure-sdk-for-ruby"  # TODO: Change in production to previous line
                repo_mask = df_out.repo_name == repo
                date_mask = df_out[date_column] == date

                if mean_dates_mask.sum() == 0:
                    cont_na += 1
                    raise ValueError
                    continue
                else:
                    real_val = df_out.loc[mean_dates_mask & repo_mask, feature].mean()
                    estimated_val = df_out.loc[date_mask & repo_mask, col].values[0]
                    if real_val - estimated_val > 0.00010:
                        raise ValueError("test_not_passed")
    if cont_na > 1:
        raise ValueError()
    logging.info("TEST PASSED")


def discard_uncompleted_windows(
    df: pd.DataFrame, evaluation_window: int, date_column: str, agg_level: str = "W"
) -> pd.DataFrame:
    """Delete all the variables with uncompleted featured enginnerinneried variables. While creating window average series, some of the data needs to be used to generate the first window variables. In particular we need #lag + #roll initial observations.

    Args:
        df (pd.DataFrame): DataFrame with new window avg. variables
        evaluation_window (int): Initial period of time required to generate window avergae varaibles.
        date_column (str): Date column name
        agg_level (str, optional): Report level for time series. Defaults to "W".

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    if agg_level == "W":
        discarted_observation = timedelta(days=evaluation_window * 7)
    elif agg_level == "D":
        discarted_observation = timedelta(days=evaluation_window)
    elif agg_level == "M":
        discarted_observation = timedelta(days=evaluation_window * 30)
    else:
        raise ValueError("Select a valid agg_level: D, M, W")

    max_date = df[
        date_column
    ].max()  # Set the max date as the last week of the 155 weeks window
    min_date = df[date_column].min() + discarted_observation
    df = df[(df[date_column] <= max_date) & (df[date_column] >= min_date)]
    return df


def moving_average_variables(
    df: pd.DataFrame,
    date_column: str,
    lag_list: Union[int, Tuple[int, ...]],
    rolling_list: Union[int, Tuple[int, ...]],
    aggregation_level: Optional[str] = "W",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply moving average variables to a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - date_column (str): Name of the date column.
    - lag_list (Union[int, Tuple[int, ...]]): List of lag values.
    - rolling_list (Union[int, Tuple[int, ...]]): List of rolling window sizes.
    - aggregation_level (Optional[str]): Aggregation level for grouping data (default is "W" for week).

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: DataFrames with ewm and rolling window features.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(["repo_name", date_column], ascending=False).reset_index()

    # TODO: Check aggregation_level and apply appropriate grouping

    # Apply ewm
    df_window_ewm = interact_categorical_numerical(
        df=df,
        lag_col=date_column,
        numerical_cols=["commit_count"],
        categorical_cols=["repo_name"],
        lag_list=lag_list,
        rolling_list=rolling_list,
        agg_funct="sum",
        store_name=None,
        rolling_function="ewm",
        freq=aggregation_level,
        parallel=False,
    )

    # Apply rolling mean
    df_window_mean = interact_categorical_numerical(
        df=df,
        lag_col=date_column,
        numerical_cols=["commit_count"],
        categorical_cols=["repo_name"],
        lag_list=lag_list,
        rolling_list=rolling_list,
        agg_funct="sum",
        store_name=False,
        rolling_function="rolling",
        freq=aggregation_level,
        parallel=False,
        parent_process="feature_enginnering",
    )

    return df_window_ewm, df_window_mean


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    logging.basicConfig(level=logging.INFO)
    TEST = True
    date_column = "date"
    lag_list = [2, 4, 6, 10]
    rolling_list = [2, 4, 6]
    evaluation_window = max(lag_list) + max(rolling_list) + 1

    if TEST:
        df = pd.read_csv("../data/preprocess/commit_series_expansion_test.csv")
    else:
        df = pd.read_csv("../data/preprocess/commit_series_expansion.csv")

    df["date"] = pd.to_datetime(df.date)
    max_min_date = datetime.today()
    for repo in df.repo_name.unique():
        df_repo = df[df.repo_name == repo]
        max_repo_date = df_repo.date.max()
        if max_repo_date < max_min_date:
            max_min_date = max_repo_date
    df = df[df.date <= max_min_date]

    df_window_mean, df_window_ewm = moving_average_variables(
        df, date_column, lag_list, rolling_list
    )

    df_out = df.merge(df_window_mean, on=[date_column, "repo_name"], how="inner")
    df_out = df_out.merge(df_window_ewm, on=[date_column, "repo_name"], how="inner")
    df_out = df_out.sort_values(["repo_name", date_column], ascending=True)

    df_out[date_column] = pd.to_datetime(df_out[date_column])

    # Trim data that falls outside the evaluation peroid. That we are ignoring the first m observation (EVALUATION WINDOW) since the first m records will be biased and will not contain accurate lag_window variables.
    if TEST:
        file_path = "../data/preprocess/featureengineering_test.csv"
    else:
        file_path = "../data/preprocess/featureengineering.csv"
    df_out = discard_uncompleted_windows(df_out, evaluation_window, date_column, "W")
    df_out.to_csv(file_path, index=False)
