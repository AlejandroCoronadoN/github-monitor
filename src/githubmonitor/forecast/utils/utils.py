"""Utilities used both in feature enginnering process and iterative forecast."""
import logging
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logging.basicConfig(level=logging.ERROR)


def generate_date_range(start_date, n: int, agg_level: str = "W"):
    """Generate a range of sequential dates starting from a given date.

    Parameters:
    start_date (str or datetime.date): The starting date.
    n (int): The number of sequential dates to generate.
    agg_level (str): The size of the step of each date in the series.

    Returns:
    list of numpy.datetime64: A list of sequential dates in numpy.datetime64 format.
    """
    # If start_date is a string, convert it to a datetime.date
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

    if agg_level == "W":
        date_range = [
            np.datetime64(start_date + timedelta(days=i * 7)) for i in range(n)
        ]
    elif agg_level == "D":
        date_range = [np.datetime64(start_date + timedelta(days=i)) for i in range(n)]
    elif agg_level == "M":
        date_range = [
            np.datetime64(start_date + timedelta(days=i * 30)) for i in range(n)
        ]

    else:
        raise ValueError("Select a valid agg_level: D, M, W")
    date_range = [pd.to_datetime(x) for x in date_range]
    return date_range


def get_testing_params(col: str) -> Tuple[int, int]:
    """Extracts testing parameters (roll and lag) from a given column name.

    Parameters:
    - col (str): The column name containing testing parameters in the format "rollX_lagY".

    Returns:
    Tuple[int, int]: A tuple containing the extracted roll and lag values.

    Example:
    >>> get_testing_params("feature_roll2_lag3")
    (2, 3)
    """
    params = col.split("_")
    roll = int(params[-3].replace("roll", ""))
    lag = int(params[-2].replace("lag", ""))
    return roll, lag


def _ewm_function(
    grouped_sum_lag: pd.DataFrame, new_colnames: List[str], roll: int
) -> pd.DataFrame:
    """Performs an avererage exponential mean over a set of numeric columns.

    Args:
        grouped_sum_lag (pd.DataFrame): DataFrame that contain only numerical values that will be processed by the average function.
        new_colnames (List[str]): List of columns to apply average of a window of time.
        roll (int): Time window length.

    Returns:
        pd.DataFrame: DataFrame with post-processed average variables.
    """
    temp_df = (
        grouped_sum_lag[new_colnames]
        .apply(lambda x: x.ewm(span=roll).mean())
        .reset_index(drop=True)
    )
    return temp_df


def _rolling_function(
    grouped_sum_lag: pd.DataFrame, new_colnames: List[str], roll: int
) -> pd.DataFrame:
    """Performs an avererage rolling mean over a set of numeric columns.

    Args:
        grouped_sum_lag (pd.DataFrame): DataFrame that contain only numerical values that will be processed by the average function.
        new_colnames (List[str]): List of columns to apply average of a window of time.
        roll (int): Time window length.

    Returns:
        pd.DataFrame: DataFrame with post-processed average variables.
    """
    temp_df = (
        grouped_sum_lag[new_colnames]
        .apply(lambda x: x.rolling(roll).mean())
        .reset_index(drop=True)
    )
    return temp_df


def _expanding_function(
    grouped_sum_lag: pd.DataFrame, new_colnames: List[str], roll: int
) -> pd.DataFrame:
    """Performs an avererage expenading mean over a set of numeric columns.

    Args:
        grouped_sum_lag (pd.DataFrame): DataFrame that contain only numerical values that will be processed by the average function.
        new_colnames (List[str]): List of columns to apply average of a window of time.
        roll (int): Time window length.

    Returns:
        pd.DataFrame: DataFrame with post-processed average variables
    """
    temp_df = (
        grouped_sum_lag[new_colnames]
        .apply(lambda x: x.expanding(roll).mean())
        .reset_index(drop=True)
    )
    return temp_df


def _process_with_freq(arg_list: object):
    """Process that creates the shift using frequencies. TODO: This function dind't pass the test so it was divided in two scenarios.

    Process
    -------
    1. Check if lag == 0. This is due a bug in pandas that doesn't do anything
    2. Create the shift directly using shift function
    3. Rename the columns to the fitting new names
    4. apply the Roll requested using Exponential Weighted Mean (ewm)

    Returns:
    -------
    A DataFrame with the numerical columns shifted in time, aggregated by
    roll param and weighted using ewm
    """
    lag = arg_list[0]
    roll = arg_list[1]
    numerical_cols = arg_list[2]
    group_nodate = arg_list[3]
    new_colnames = arg_list[4]
    rolling_function = arg_list[5]
    freq = arg_list[6]
    grouped_sum = arg_list[7]

    if lag == 0:
        temp_df = grouped_sum[numerical_cols + group_nodate].shift(-1, freq=freq)
        temp_df = temp_df.shift(1, freq=freq)
    else:
        temp_df = grouped_sum[numerical_cols + group_nodate].shift(lag, freq=freq)
    temp_df = temp_df.rename(columns=dict(zip(numerical_cols, new_colnames)))
    temp_df = temp_df.reset_index()
    grouped_sum_lag = temp_df.groupby(group_nodate)
    # ------------------------------Roll---------------------------
    if rolling_function == "ewm":
        temp_df[new_colnames] = _ewm_function(grouped_sum_lag, new_colnames, roll)
    elif rolling_function == "rolling":
        temp_df[new_colnames] = _rolling_function(grouped_sum_lag, new_colnames, roll)
    elif rolling_function == "expanding":
        temp_df[new_colnames] = _expanding_function(grouped_sum_lag, new_colnames, roll)
    else:
        raise Exception(
            "'rolling_function {} is not implemented".format(rolling_function)
        )

    return temp_df


def interact_categorical_numerical(
    df: pd.DataFrame,
    lag_col: str,
    numerical_cols: List[str],
    categorical_cols: List[StopAsyncIteration],
    lag_list: List[int],
    rolling_list: List[int],
    agg_funct: str = "sum",
    rolling_function: str = "ewm",
    freq=None,
    group_name: str = None,
    store_name: bool = False,
    parallel: bool = True,
    parent_process: str = "feature_enginnering",
):
    """Function that takes a list of categorical values and generates all the rolling averages for each lag and window period given.

    Args:
        df (pd.DataFrame): Data to be processed
        lag_col (str): Column name for the lag (usually the date column)
        numerical_cols (List[str]): Column names for the value from where the metrics are calculated
        categorical_cols (List[StopAsyncIteration]): List containing the names of the categorical cols to be grouped by
        lag_list (List[int]): List containing each lag to be applied (only for date data).
        This lag is applied with a pandas shift.
        rolling_list (List[int]): List containing each rolling list to be applied.
        This is the number of values that will calculate the metric.
        If values are less than this number, a NaN is reported.
        agg_funct (str, optional): Current functions supported are only 'sum' and 'count', which are
        the aggregator functions executed over numerical_cols.. Defaults to "sum".
        rolling_function (str, optional): Rolling window calculations.
            rolling: equal weghted
            ewm: exponential weighted.
            expanding: expanding weight. Defaults to "ewm".
        freq (_type_, optional): If none, no transformation is done during group by
            Frequency capital letters as stated here:
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
            Setting freq to the observations periodicity it's encourage, otherwise
            if some value it's no reported at a particular date the function will
            bot consider this date for the aggregation process. Defaults to None.
        group_name (str, optional): base name for the columns creator. If none given, it will make a long identifiable name. Defaults to None.
        store_name (bool, optional): If True, the resulting cols will only have the original colname
        If False, a structured history-keeping name is given. Defaults to False.
        parallel (bool, optional): Use different number of cores depending on the value. Defaults to True.
        parent_process (str, optional): Which script is calling this feature. Defaults to "feature_enginnering".

    Raises:
        Exception: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if not group_name:
        group_name = "{cat_cols}".format(cat_cols="_".join(categorical_cols))

    group_nodate = list(set(categorical_cols) - {lag_col})
    categorical_cols = group_nodate + [lag_col]

    if freq is not None:
        group_by_arg = group_nodate + [pd.PeriodIndex(df[lag_col], freq=freq)]
    else:
        group_by_arg = categorical_cols

    grouped = df.groupby(group_by_arg)
    t_ser = grouped[numerical_cols]

    if agg_funct == "sum":
        t_ser = t_ser.sum().sort_index()
    elif agg_funct == "count":
        t_ser = t_ser.count().sort_index()
    else:
        raise Exception("'{0}' is not implemented".format(agg_funct))

    t_ser = t_ser.reset_index()

    if freq:
        t_ser[lag_col] = t_ser[lag_col].apply(lambda x: x.to_timestamp())
        grouped_sum = t_ser.set_index(lag_col)
        t_set = t_ser[categorical_cols].head(0)
    else:
        grouped_sum = t_ser.groupby(group_nodate)
        t_set = t_ser[categorical_cols]

    col_naming = (
        "{col}"
        if store_name
        else "{funct}_{group}_on_{col}_with_{process}_{rolling_function}"
    )
    thread_params = []
    for lag in lag_list:
        for roll in rolling_list:
            process = "roll{n_roll}_lag{n_lag}".format(n_roll=roll, n_lag=lag)
            new_colnames = [
                col_naming.format(
                    funct=agg_funct,
                    group=group_name,
                    col=col,
                    process=process,
                    rolling_function=rolling_function,
                )
                for col in numerical_cols
            ]
            logging.info("new_cols: {0}".format(", ".join(new_colnames)))

            if len(new_colnames) > 0:
                thread_params.append(
                    (
                        lag,
                        roll,
                        numerical_cols,
                        group_nodate,
                        new_colnames,
                        rolling_function,
                        freq,
                        grouped_sum,
                    )
                )
    if parallel:
        if parent_process == "feature_enginnering":
            pool = Pool(cpu_count() - 1)
        elif parent_process == "prediction":
            pool = Pool(6)  # Tune this parameter with htop and measure time
    else:
        pool = Pool(1)
    temp_dfs = pool.map(_process_with_freq, thread_params)
    for each_df in temp_dfs:
        t_set = t_set.merge(each_df, how="outer", on=categorical_cols)

    if (lag_col in t_set.columns) & (lag_col not in categorical_cols):
        t_set.drop(lag_col, axis=1, inplace=True)

    pool.close()

    return t_set


def perform_standardization(
    data_frame: pd.DataFrame, column_name: str, method: str = "z-score"
) -> Tuple[pd.DataFrame, object]:
    """Perform standardization on a column in a DataFrame.

    Parameters:
    data_frame (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to standardize.
    method (str): The method of standardization. Options: 'z-score', 'min-max', 'robust'.

    Returns:
    pandas.DataFrame: DataFrame with the standardized column.
    """
    if method == "z-score":
        scaler = StandardScaler()
    elif method == "min-max":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(
            "Invalid standardization method. Choose from 'z-score', 'min-max', or 'robust'"
        )

    data_frame.loc[:, column_name] = scaler.fit_transform(
        data_frame[column_name].values.reshape(-1, 1)
    )
    return data_frame, scaler


# Define the missing functions
def print_grid_scores(grid_search: object):
    """Print the results for all the models trained in the grid_search algorithm.

    Args:
        grid_search (object): trainned models by scikit learn.
    """
    # Access the cross-validation results
    results = grid_search.cv_results_
    # Print the results
    for mean_score, std_score, params in zip(
        results["mean_test_score"], results["std_test_score"], results["params"]
    ):
        logging.info(
            f"Mean Score: {mean_score:.4f}, Std Dev: {std_score:.4f}, Params: {params}"
        )


def current_date_formated() -> str:
    """Takes the current date and pass it as str yyyymmdd.

    Returns:
        str: yyyymmdd format
    """
    # Get the current date and time
    current_datetime = datetime.today()

    # Convert to string with the format YYYYMMDD
    formatted_date = current_datetime.strftime("%Y%m%d")
    return formatted_date


# Data Preprocessing Functions
def preprocess_data(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Additional control to pass only variables available in hyperparamter optimization and iterative prediction scripts.

    Args:
        df (pd.DataFrame): Initital DataFrame
        date_col (str): date column name

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    selected_cols = ["repo_name"]
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    for col in df.columns:
        if "_lag" not in col and col != "repo_name":
            selected_cols.append(col)
        elif "_lag" in col:
            lag = col.split("_")[8]
            lag = int(lag.replace("lag", ""))
            selected_cols.append(col)

    return df[selected_cols]


def weekly_group(
    df: pd.DataFrame, date_col: str, index_cols: List[str], sum_cols: List[str], mean_cols: List[str]
) -> pd.DataFrame:
    """
    Perform weekly grouping of the DataFrame based on specified columns.

    The function groups the DataFrame by the specified 'date_col' and 'index_cols',
    calculating the sum and mean for the provided columns 'sum_cols' and 'mean_cols', respectively.
    It also checks for consistency between the sum and mean values for validation purposes.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Name of the column containing date information.
        index_cols (List[str]): List of column names to be used as index for grouping.
        sum_cols (List[str]): List of column names for which the sum will be calculated.
        mean_cols (List[str]): List of column names for which the mean will be calculated.

    Returns:
        pd.DataFrame: DataFrame with weekly grouped data.

    Raises:
        ValueError: If there is inconsistency between the sum and mean values.

    Example:
        ```
        df_weekly_grouped = weekly_group(df, 'Date', ['Repository'], ['Column1'], ['Column2'])
        ```

    Note:
        Ensure that the 'date_col' and 'index_cols' are present in the DataFrame.
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
            raise ValueError("Inconsistency between sum values.")
    else:
        total_test_2 = df_out[test_col].mean()
        if np.abs(total_test_1 - total_test_2) / total_test_1 > 0.2:
            raise ValueError("Inconsistency between mean values.")

    df_out[date_col] = pd.to_datetime(df_out[date_col])
    df_out = df_out.sort_values(index_cols + [date_col], ascending=True)
    return df_out



def convert_to_first_monday_of_week(input_date):
    """Convert any date to the date linked to the first Monday of the same week.

    Parameters:
    input_date (str or datetime.date): The input date to convert.

    Returns:
    datetime.date: The date linked to the first Monday of the same week.
    """
    input_date = pd.to_datetime(input_date)
    # Calculate the start of the week (Monday)
    start_of_week = input_date - pd.DateOffset(days=input_date.weekday())
    return start_of_week.date()
