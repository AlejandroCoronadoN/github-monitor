"""Complete the time series DataFrame by reporting all the weeks."""

import logging
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
# Append the 'src' directory to the Python path
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(src_dir)

logging.basicConfig(level=logging.INFO)


def create_seasonal_controls(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Create year, month and week variables.

    Args:
        df (pd.DataFrame): data frame at a weekly level
        date_column (str): Name of date column

    Returns:
        pd.DataFrame: DataFrame with new seasonal variables
    """
    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["week"] = df[date_column].dt.strftime("%U").astype(int)
    return df


def expand_time_series(
    df: pd.DataFrame,
    date_column: str,
    df_index: pd.DataFrame,
    df_dates_week: pd.DataFrame,
) -> pd.DataFrame:
    """Expansion is being done at a weekly level. If a repository only report commit in 3 weeks, this function will create 257 new entries with a commit_count equal to 0, representing all the week were the repository existed but no commit was made.

    Args:
        df (pd.DataFrame): Repository commit count at a weekly level (only reported weeks).
        date_column (str): Name of the date column
        df_index (pd.DataFrame): allows us to keep track of the different repositories
        df_dates_week (pd.DataFrame): Complete series of weeks from start_date to end_date reported at a weekly level.

    Raises:
        ValueError: Internal control that report information for a 5 years long evaluation period

    Returns:
        pd.DataFrame: DataFrame with 260 week datapoints for each repository. All repositories share the same dates.
    """
    df_all = pd.DataFrame()
    for i in tqdm(range(len(df_index))):
        repo_id = df_index.loc[i, "repo_name"]
        df_repo = df[(df["repo_name"] == repo_id)]
        df_dates_week["repo_name"] = repo_id
        df_repo[date_column] = pd.to_datetime(df_repo[date_column])
        # MERGE
        df_expand = df_repo.merge(
            df_dates_week, on=["repo_name", date_column], how="outer"
        )
        if len(df_all) == 0:
            df_all = df_expand
        else:
            df_all = pd.concat([df_all, df_expand], ignore_index=True)
            if i % 1000 == 0:  # Checkpoint for comple data
                df_all.to_csv("DF_AUX_EXPAND_{i}.csv", index=False)
    return df_all


def impute_default(df: pd.DataFrame, cols: list, val: float) -> pd.DataFrame:
    """Repalace non reported or undefined values for a values pased by the user Ex: 0.

    Args:
        df (pd.DataFrame): Initial DataFrame with potenin nan values
        cols (list): List of columns were we want to input val
        val (float): Value to be imputed

    Returns:
        pd.DataFrame: DataFrame with replace nan values
    """
    for col in cols:
        df[col] = df[col].replace(np.nan, val)
    return df


def create_weekly_date_dataframe(start_date, end_date, week_start="sunday"):
    """Create a DataFrame with a date series grouped by week (first day of each week, Sunday).

    Parameters:
    start_date (str): Start date in YYYY-MM-DD format.
    end_date (str): End date in YYYY-MM-DD format.

    Returns:
    pandas.DataFrame: DataFrame with a date series.
    """
    if week_start == "sunday":
        date_range = pd.date_range(start=start_date, end=end_date, freq="W-SUN")
        date_df = pd.DataFrame({"date": date_range})
    elif week_start == "monday":
        date_range = pd.date_range(start=start_date, end=end_date, freq="W-MON")
        date_df = pd.DataFrame({"date": date_range})
    return date_df


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    TEST = True
    date_column = "date"
    aggregation_cols = ["year", "month", "week"]

    if TEST:
        df = pd.read_csv("../data/preprocess/commit_history_subset_test.csv")
    else:
        df = pd.read_csv("../data/preprocess/commit_history_subset.csv")

    group_id = ["repo_name"]

    df_all = pd.DataFrame()
    df[date_column] = pd.to_datetime(df[date_column])
    df_index = df.groupby(group_id).first().reset_index()

    start_date = df[date_column].min()
    end_date = df[date_column].max()

    df_dates_week = create_weekly_date_dataframe(
        start_date, end_date, week_start="sunday"
    )  # Choose between sunday or monday
    df_expand = expand_time_series(df, date_column, df_index, df_dates_week)
    df_expand = create_seasonal_controls(df_expand, date_column="date")
    df_all_preproc = impute_default(df_expand, ["commit_count"], 0)
    df_all_preproc = df_all_preproc[
        ["repo_name", "year", "commit_count", "date", "month", "week"]
    ]
    if TEST:
        df_all_preproc.to_csv(
            "../data/preprocess/commit_series_expansion_test.csv",
            index=False,
        )
    else:
        df_all_preproc.to_csv(
            "../data/preprocess/commit_series_expansion.csv",
            index=False,
        )
