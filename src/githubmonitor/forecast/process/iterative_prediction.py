"""Create sequential predictions over our data using previous predictions to make new predictions about more distanta periods of time (weeks ahead into the future)."""

import concurrent
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

# TODO: Ruff is forcing us to define the following instructions after the import. For testing this script separately rearange the order of these lines of code.
from forecast.utils.utils import (
    interact_categorical_numerical,
    perform_standardization,
    preprocess_data,
)
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
# Append the 'src' directory to the Python path
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(src_dir)
logging.basicConfig(level=logging.INFO)


def rename_predictions(df: pd.DataFrame, df_prediction: pd.DataFrame) -> pd.DataFrame:
    """Rename the dataframe with more legible names and allows us to compare predictions with the actual reported values.

    Args:
        df (pd.DataFrame): DataFramse with actual values of the target variable.
        df_prediction (pd.DataFrame): Predictions.

    Returns:
        pd.DataFrame: DataFrame with both actual and forecasted values.
    """
    if len(df) > 0:
        df = df.rename(
            columns={
                "commit_count": "Commit Forecast",
            }
        )

        df_original_prediction = df.merge(
            df_prediction[["commit_count", "date", "repo_name"]],
            on=["repo_name", "date"],
        )

        df_original_prediction = df_original_prediction.rename(
            columns={
                "commit_count": "Commit Real",
                "date": "Date",
                "repo_name": "Repository",
            }
        )

        return df_original_prediction
    else:
        return df


def update_with_predictions(
    df_model_input: pd.DataFrame,
    df_iter: pd.DataFrame,
    df_index: pd.DataFrame,
    df_predicted: pd.DataFrame,
    predicted_date: datetime,
    scalers_dict: dict,
    model_loaded: object,
    model_features,
) -> pd.DataFrame:
    """Reinserts the prediction as commit_count (target variable), allowing us to take these values as new entries and reprocess data with interact_categorical_numerical and prepare the DataFrame for the next round of predictions (t+1).

    Args:
        df_model_input (pd.DataFrame): Data points to be passed to the model and make a new forecast.
        df_iter (pd.DataFrame): Previous Data with past information. commit_count is updated with predicted values.
        df_index (pd.DataFrame): Keeps track of the dates we want to make forecast.
        df_predicted (pd.DataFrame): DataFrame with previous predictions
        predicted_date (datetime): Week to be predicted in current iteration.
        scalers_dict (dict): Dictionary with standarization models for each column
        model_loaded (object): Model to make predictions (XGbooost, LightGBM, RandomForest)

    Returns:
        pd.DataFrame: Updated DataFrame
    """
    df_result = df_iter.copy()  # * UN-STANDARDIZED
    single_prediction_mask = df_index.date == predicted_date
    # Use the loaded model to make predictions
    model = model_loaded
    df_model_target = df_model_input.copy()  #! STANDARDIZED
    # Standardize Values
    for col in df_model_target:
        if col in scalers_dict.keys():
            scaler = scalers_dict[col]
            df_model_target[col] = scaler.transform(
                df_model_target[col].values.reshape(-1, 1)
            )

    X_test_imputed = df_model_target.replace(np.nan, 0)
    X_test_imputed = X_test_imputed[model_features]
    predictions = model.predict(X_test_imputed)
    scaler = scalers_dict["commit_count"]  # ? TRANSFORM STANDARDIZED -> UN-STANDARDIZED
    destandardize_value = scaler.inverse_transform(
        predictions.reshape(-1, 1)
    )  # * UN-STANDARDIZED

    df_result["commit_count"] = destandardize_value  # * UN-STANDARDIZED

    # Selecting one row, we need to previous information for creating
    df_result = df_result[single_prediction_mask]
    df_result = df_result[df_predicted.columns]
    df_predicted = pd.concat(
        [df_predicted, df_result], ignore_index=True
    )  # * UN-STANDARDIZED * #* UN-STANDARDIZED

    return df_predicted  # * UN-STANDARDIZED


def compare_merge(
    df_index: pd.DataFrame, df_lag: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare and merge two DataFrames based on repository and date.

    This function ensures the existence of matching repository and date combinations in both DataFrames.
    If a repository or date is missing in one of the DataFrames, it is deleted from both DataFrames.

    Args:
        df_index (pd.DataFrame): DataFrame containing the main data.
        df_lag (pd.DataFrame): DataFrame containing lagged or secondary data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames after ensuring matching repository and date combinations.

    Raises:
        ValueError: If a repository-date combination is missing in either df_index or df_lag.

    Example:
        ```
        df_main = read_main_data()
        df_lag = read_lagged_data()
        df_main, df_lag = compare_merge(df_main, df_lag)
        ```

    Note:
        Ensure that 'repo_name' and 'date' columns are present in the DataFrames.
    """
    for repo in df_index.repo_name.unique():
        df_index_repo = df_index[df_index.repo_name == repo]
        df_lag_repo = df_lag[df_lag.repo_name == repo]

        if len(df_index_repo) == 0:  # is empty
            df_index = df_index[df_index.repo_name != repo]
            df_lag = df_lag[df_lag.repo_name != repo]
            logging.warning(f"Repository deleted: {repo} ")

        for date in df_index_repo.date.unique():
            df_index_repo_date = df_index_repo[df_index_repo.date == date]
            df_lag_repo_date = df_lag_repo[df_lag_repo.date == date]
            if len(df_index_repo_date) == 0:
                raise ValueError("repo - date not found in lag")
                df_index = df_index[df_index.date != date]
                df_lag = df_index[df_lag.date != date]
                logging.warning(f"Date deleted: {repo} ")

    for repo in df_lag.repo_name.unique():
        df_index_repo = df_index[df_index.repo_name == repo]
        df_lag_repo = df_lag[df_lag.repo_name == repo]
        if len(df_lag_repo) == 0:
            df_index = df_index[df_index.repo_name != repo]
            df_lag = df_index[df_lag.repo_name != repo]
            logging.warning(f"Repository deleted: {repo} ")

        for date in df_lag_repo.date.unique():
            df_index_repo_date = df_index_repo[df_index_repo.date == date]
            df_lag_repo_date = df_lag_repo[df_lag_repo.date == date]

            if len(df_lag_repo_date) == 0:  # is empty
                raise ValueError("repo - date not found in lag")
                df_index = df_index[df_index.date != date]
                df_lag = df_index[df_lag.date != date]
                logging.warning(f"Date deleted: {repo} ")

    return df_index, df_lag


def filter_dates(
    df: pd.DataFrame, selected_cols: List[str], forecast_start, min_date
) -> pd.DataFrame:
    """Filter DataFrame columns and dates.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.
        selected_cols (List[str]): List of columns to include in the prediction.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Data Preprocessing
    df_predicted = df.copy()
    # Give minimal space for the interact_categoprical_numerical function to perform operations (increase speed)
    df_predicted = df_predicted[
        (df_predicted.date >= min_date) & (df_predicted.date < forecast_start)
    ]
    df_predicted = df_predicted[selected_cols]
    return df_predicted


def make_iterative_prediction(
    df: pd.DataFrame,
    model: str,
    scalers_dict: dict,
    lag_list: List[int],
    rolling_list: List[int],
    forecast_start,
    parallel,
    min_date,
    evaluation_window,
    model_features,
    prediction_window,
    production=False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Crete iterative prediction by passing the initial Data and the updating the DataFrame with new predictions. The model is feeded every iteration with a new set of average window and lag variables. Each model makes it's own predictions over a time frame = PREDICTION_WINDOW.

    Args:
        df (pd.DataFrame): Initial DataFrame
        model (str): Name of the model to be loaded (xgboost, lgbm, randomforest)
        scalers_dict (dict): dictionary with scaler models.
        lag_list (List[int]): List of lag variables to be passed to interact_categorical_numerical
        rolling_list (List[int]): List of average window variables to be passed to interact_categorical_numerical

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrame with predictions and evaluation of the model at 1, 2 and 3 months forecast.
    """
    selected_cols = [
        "repo_name",
        "date",
        "commit_count",
    ]
    # Data Preprocessing

    df_1month_validtaion = pd.DataFrame()
    df_2month_validtaion = pd.DataFrame()
    df_3month_validtaion = pd.DataFrame()
    if production:
        model_filename = (
            f"githubmonitor/forecast/process/models/commit_count_{model}.joblib"
        )
    else:
        model_filename = f"./models/commit_count_{model}.joblib"
    model_loaded = joblib.load(model_filename)

    df_predicted = filter_dates(
        df,
        forecast_start=forecast_start,
        selected_cols=selected_cols,
        min_date=min_date,
    )
    for i in tqdm(range(prediction_window)):
        prediction_date = df_predicted.date.max() + timedelta(days=7)

        upper_date = df_predicted.date.max()  # * UN-STANDARDIZED
        lower_date = prediction_date - timedelta(days=(evaluation_window * 7))
        last_obs = df_predicted[df_predicted.date == upper_date]
        df_iteration = df_predicted.copy()  # * UN-STADARDIZED
        df_iteration = df_iteration[df_iteration.date >= lower_date]

        # Prepare training and testing sets for XGBoost models
        new_entry, predicted_date = create_model_entry(last_obs=last_obs)
        for repo in df_iteration.repo_name.unique():
            new_entry_repo = new_entry.copy()
            new_entry_repo["repo_name"] = repo
            df_iteration = pd.concat([df_iteration, new_entry_repo], ignore_index=True)

        df_window_ewm = interact_categorical_numerical(
            df=df_iteration,  # * UN-STADARDIZED
            lag_col="date",
            numerical_cols=["commit_count"],
            categorical_cols=["repo_name"],
            lag_list=lag_list,
            rolling_list=rolling_list,
            agg_funct="sum",
            store_name=False,
            rolling_function="ewm",
            freq="W",
            parallel=True,
        )
        df_window_mean = interact_categorical_numerical(
            df=df_iteration,  # * UN-STADARDIZED
            lag_col="date",
            numerical_cols=["commit_count"],
            categorical_cols=["repo_name"],
            lag_list=lag_list,
            rolling_list=rolling_list,
            agg_funct="sum",
            store_name=False,
            rolling_function="rolling",
            freq="W",
            parallel=True,
        )

        df_index = df_iteration[["date", "repo_name"]].copy()
        df_index = df_index.merge(df_window_mean, on=["date", "repo_name"], how="left")
        df_index = df_index.merge(df_window_ewm, on=["date", "repo_name"], how="left")

        df_index = df_index[
            (df_index.date >= lower_date) & (df_index.date <= prediction_date)
        ]
        features = [x for x in df_index.columns if x not in ["date", "repo_name"]]
        df_index = df_index.sort_values(["repo_name", "date"], ascending=True)
        df_model_input = df_index[features]
        df_predicted = update_with_predictions(
            df_model_input=df_model_input,
            df_iter=df_iteration,
            df_index=df_index,
            df_predicted=df_predicted,
            scalers_dict=scalers_dict,
            predicted_date=predicted_date,
            model_loaded=model_loaded,
            model_features=model_features,
        )

        df_entry = df_predicted[df_predicted.date == predicted_date]
        df_real = df[(df.date == predicted_date)][["commit_count"]]
        if not parallel:
            logging.info(
                f"\n \ndate: {predicted_date}\nMODEL: {model}  \n\t ** PREDICTED : \n {df_entry[['commit_count']]} \n\n\t ** ORIGINAL :\n {df_real}\n"
            )

        if i != 0:
            if i in range(
                2, 6
            ):  # ? ONE month ARROUND i ==4 (1 Month window validation)
                if len(df_1month_validtaion) == 0:
                    df_1month_validtaion = df_entry.copy()
                else:
                    df_1month_validtaion = pd.concat([df_1month_validtaion, df_entry])

            if i in range(6, 10):  # ? TWO months 8 weeks
                if len(df_2month_validtaion) == 0:
                    df_2month_validtaion = df_entry.copy()
                else:
                    df_2month_validtaion = pd.concat([df_2month_validtaion, df_entry])

            if i in range(10, 14):  # ? THREE months 12 weeks
                if len(df_3month_validtaion) == 0:
                    df_3month_validtaion = df_entry.copy()
                else:
                    df_3month_validtaion = pd.concat([df_3month_validtaion, df_entry])
    return (
        df_predicted,
        df_1month_validtaion,
        df_2month_validtaion,
        df_3month_validtaion,
    )


def parallel_retrive(
    results: list, df_prediction: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Allows us to retrive the results from each core when executing parallel processing.

    Args:
        results (list): A list that contains the output of make_iterative_prediction executed for each repository.
        df_prediction (pd.DataFrame): Data  with predcition.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrame with predictions and evaluation of the model at 1, 2 and 3 months forecast.
    """
    df_1month_validtaion_all = pd.DataFrame()
    df_2month_validtaion_all = pd.DataFrame()
    df_3month_validtaion_all = pd.DataFrame()
    df_predicted_all = pd.DataFrame()

    for result in results:
        (
            df_predicted,
            df_1month_validtaion_branch,
            df_2month_validtaion_branch,
            df_3month_validtaion_branch,
        ) = result.result()

        if len(df_predicted_all) == 0:
            df_predicted_all = rename_predictions(df_predicted, df_prediction)
        else:
            df_predicted = rename_predictions(df_predicted, df_prediction)
            df_predicted_all = pd.concat([df_predicted_all, df_predicted])
        if len(df_1month_validtaion_branch) > 0:
            if len(df_1month_validtaion_all) == 0:
                df_1month_validtaion_all = rename_predictions(
                    df_1month_validtaion_branch, df_prediction
                )
            else:
                df_1month_validtaion_branch = rename_predictions(
                    df_1month_validtaion_branch, df_prediction
                )
                df_1month_validtaion_all = pd.concat(
                    [df_1month_validtaion_all, df_1month_validtaion_branch]
                )

            if len(df_2month_validtaion_all) == 0:
                df_2month_validtaion_all = rename_predictions(
                    df_2month_validtaion_branch, df_prediction
                )
            else:
                df_2month_validtaion_branch = rename_predictions(
                    df_2month_validtaion_branch, df_prediction
                )
                df_2month_validtaion_all = pd.concat(
                    [df_2month_validtaion_all, df_2month_validtaion_branch]
                )

            if len(df_3month_validtaion_all) == 0:
                df_3month_validtaion_all = rename_predictions(
                    df_3month_validtaion_branch, df_prediction
                )
            else:
                df_3month_validtaion_branch = rename_predictions(
                    df_3month_validtaion_branch, df_prediction
                )
                df_3month_validtaion_all = pd.concat(
                    [df_3month_validtaion_all, df_3month_validtaion_branch]
                )

    return (
        df_predicted_all,
        df_1month_validtaion_all,
        df_2month_validtaion_all,
        df_3month_validtaion_all,
    )


def create_model_entry(last_obs: pd.DataFrame) -> Tuple[pd.DataFrame, datetime]:
    """Converts the prediction into a valid DataFrame row that can be added to the previous DataFrame.

    Args:
        last_obs (pd.DataFrame): Data entry associated with the maxium date in the TimeSeries.

    Returns:
        Tuple[pd.DataFrame, datetime]: New DataFrame and the last date reported.
    """
    date = last_obs["date"] + timedelta(days=7)
    new_obs_dict = {
        "date": date.values[0],
        "commit_count": 0,
    }
    df_new = pd.DataFrame(new_obs_dict, index=[date.index[0]])
    return df_new, date.values[0]


def get_scalers(df: pd.DataFrame) -> dict:
    """Standardize each column and saves the scaler in the scalers_dict.

    Args:
        df (pd.DataFrame): Each column of the DataFrame will be standardized.

    Returns:
        dict: Scalers
    """
    scalers_dict = {}
    df_prediction = df.copy()
    for col in df_prediction.columns:
        if col not in [
            "date",
            "repo_name",
        ]:
            df_prediction, scaler = perform_standardization(
                df_prediction, col, method="z-score"
            )
            scalers_dict[col] = scaler
    return scalers_dict


def create_iterative_forecast(
    df: pd.DataFrame,
    model,
    model_features,
    forecast_start,
    lag_list,
    rolling_list,
    prediction_window,
    min_date,
    evaluation_window,
    parallel=True,
    production=False,
) -> pd.DataFrame:
    """Base function that handles the iterative prediction model.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Output DataFrame.
    """
    df["date"] = pd.to_datetime(df.date)
    scalers_dict = get_scalers(df)
    if parallel:
        with ProcessPoolExecutor() as executor:
            results = []
            df_prediction = df.copy()
            result = executor.submit(
                make_iterative_prediction,
                df=df_prediction,
                model=model,
                scalers_dict=scalers_dict,
                lag_list=lag_list,
                rolling_list=rolling_list,
                forecast_start=forecast_start,
                parallel=parallel,
                min_date=min_date,
                evaluation_window=evaluation_window,
                production=production,
                prediction_window=prediction_window,
                model_features=model_features,
            )
            results.append(result)
            concurrent.futures.wait(results)
            # Retrieve the results
            (
                df_predicted_all,
                df_1month_validtaion_all,
                df_2month_validtaion_all,
                df_3month_validtaion_all,
            ) = parallel_retrive(
                results=results,
                df_prediction=df,
            )
    else:  # Helps debugging the application.
        df_predicted_all = pd.DataFrame()
        df_prediction = df.copy()
        scalers_dict = scalers_dict
        (
            df_predicted,
            df_1month_validtaion_branch,
            df_2month_validtaion_branch,
            df_3month_validtaion_branch,
        ) = make_iterative_prediction(
            df_prediction,
            model,
            scalers_dict=scalers_dict,
            lag_list=lag_list,
            rolling_list=rolling_list,
            forecast_start=forecast_start,
            parallel=parallel,
            min_date=min_date,
            evaluation_window=evaluation_window,
            production=production,
            prediction_window=prediction_window,
            model_features=model_features,
        )

        if len(df_predicted_all) == 0:
            df_predicted_all = rename_predictions(df_predicted, df_prediction)
        else:
            df_predicted_renamed = rename_predictions(df_predicted, df_prediction)
            df_predicted_all = pd.concat([df_predicted, df_predicted_renamed])
    #! IMPORTANT -  PART of the forecast.
    if production:
        df_out_production = df_predicted.rename(
            columns={
                "repo_name": "Repository",
                "date": "Date",
                "commit_count": "Commit Forecast",
            }
        )
        df_out_production["Commit Real"] = 0
        df_out_production.to_csv(
            f"githubmonitor/forecast/data/process/predictions_{model}_production.csv"
        )
        # Date bigger than toda
        return df_out_production

    else:
        df_predicted_all.to_csv(f"../data/process/predictions_{model}.csv")
        return df_predicted_all


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    TEST = True
    parallel = True
    prediction_contained = True
    retrain_models = True
    lag_list = [2, 4, 6, 10]
    rolling_list = [2, 4, 6]
    cut_date = pd.to_datetime("2021-12-26")
    evaluation_window = max(lag_list) + max(rolling_list) + 1
    # ? Note: that we are going to be placed at t='2021-12-26'  the we need to start the iteration at MAX date to cover predictions until '2021-12-26'. Notice also that you will need at least evaluation_window observations before max date in order to create predictions for forecast_start. The code advance one week at a time to recalculate the predictions using previous predictions.

    model_features = [
        "sum_repo_name_on_commit_count_with_roll2_lag2_ewm",
        "sum_repo_name_on_commit_count_with_roll4_lag2_ewm",
        "sum_repo_name_on_commit_count_with_roll6_lag2_ewm",
        "sum_repo_name_on_commit_count_with_roll2_lag4_ewm",
        "sum_repo_name_on_commit_count_with_roll4_lag4_ewm",
        "sum_repo_name_on_commit_count_with_roll6_lag4_ewm",
        "sum_repo_name_on_commit_count_with_roll2_lag6_ewm",
        "sum_repo_name_on_commit_count_with_roll4_lag6_ewm",
        "sum_repo_name_on_commit_count_with_roll6_lag6_ewm",
        "sum_repo_name_on_commit_count_with_roll2_lag10_ewm",
        "sum_repo_name_on_commit_count_with_roll4_lag10_ewm",
        "sum_repo_name_on_commit_count_with_roll6_lag10_ewm",
        "sum_repo_name_on_commit_count_with_roll2_lag2_rolling",
        "sum_repo_name_on_commit_count_with_roll4_lag2_rolling",
        "sum_repo_name_on_commit_count_with_roll6_lag2_rolling",
        "sum_repo_name_on_commit_count_with_roll2_lag4_rolling",
        "sum_repo_name_on_commit_count_with_roll4_lag4_rolling",
        "sum_repo_name_on_commit_count_with_roll6_lag4_rolling",
        "sum_repo_name_on_commit_count_with_roll2_lag6_rolling",
        "sum_repo_name_on_commit_count_with_roll4_lag6_rolling",
        "sum_repo_name_on_commit_count_with_roll6_lag6_rolling",
        "sum_repo_name_on_commit_count_with_roll2_lag10_rolling",
        "sum_repo_name_on_commit_count_with_roll4_lag10_rolling",
        "sum_repo_name_on_commit_count_with_roll6_lag10_rolling",
    ]

    if prediction_contained:
        if retrain_models:
            file_path = "../data/preprocess/featureengineering_test.csv"
            df_test = pd.read_csv(file_path)
            df_test["date"] = pd.to_datetime(df_test["date"])
            min_date = df_test.date.min()
            prediction_window = 104  # two years
            forecast_start = cut_date - timedelta(days=(prediction_window * 7))

        else:
            prediction_window = 12  # Three months
            forecast_start = cut_date - timedelta(days=(prediction_window * 7))
            min_date = forecast_start - timedelta(days=(53 * 7))

    else:
        # Start from the last day and starts making iterations over the future
        forecast_start = cut_date
        min_date = forecast_start - timedelta(days=(53 * 7))

    for model in ["xgboost", "randomforest"]:
        start_time = time.time()
        if TEST:
            file_path = "../data/preprocess/featureengineering_test.csv"
        else:
            file_path = "../data/preprocess/featureengineering.csv"

        df = pd.read_csv(file_path)
        df = df[~df["commit_count"].isna()]
        # Data Preprocessing
        df = preprocess_data(df, "date")  # 30 days
        df = pd.read_csv(file_path)
        df_predicted_all = create_iterative_forecast(
            df,
            model=model,
            forecast_start=forecast_start,
            lag_list=lag_list,
            prediction_window=prediction_window,
            rolling_list=rolling_list,
            parallel=parallel,
            evaluation_window=evaluation_window,
            min_date=min_date,
            model_features=model_features,
        )

        end_time = time.time()
        elapsed_time = round(end_time - start_time)

        logging.info(f"Time taken for the operation: {elapsed_time} seconds")
        text_file = open(f"logs_{model}_November.txt", "w")
        log_string = f"""
        MODEL: {model}
        PROCESSING TIME:  {elapsed_time}

        PARALLEL: {parallel}

        CORES: 32
        SUBPROCESS CORES: 6
        """
        text_file.write(log_string)

        # close file
        text_file.close()
