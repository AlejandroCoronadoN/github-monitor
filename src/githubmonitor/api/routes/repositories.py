"""Repositorie routes.

This module contains the forecast process of the application. by passing the historic information of a Github repository, this module uses create_forecast function to call pretrainned models and functions that live in the features and models directory of this project. These functions have been used previously to test and tune our models. This is the production version of the forecast algorithm and We are only making predictions and testing them trough the github_forecast endpoint.

"""

import logging
import os
import time
from datetime import timedelta

import githubmonitor.api.schemas
import joblib
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from githubmonitor.forecast.preprocess.feature_engineering import (
    moving_average_variables,
)
from githubmonitor.forecast.process.forecast_ensemble import (
    create_forecast_horizon,
    create_month_dummy,
    prepare_all_models,
    set_cluster,
    test_uniqueness_branch_date,
    train_elasticnet_ensemble_model,
)
from githubmonitor.forecast.process.iterative_prediction import (
    create_iterative_forecast,
)

relative_path = os.path.dirname(os.path.realpath(__file__)).replace(
    "githubmonitor/api/routes/", ""
)
relative_path = relative_path.replace("githubmonitor/api/routes", "")
logging.info(relative_path)

model_path = os.path.join(
    relative_path,
    "githubmonitor/forecast/process/models/commit_count_randomforest.joblib",
)
MODEL = joblib.load(model_path)

logging.basicConfig(level=logging.INFO)
# Router basic config
repositories_router = APIRouter(
    prefix="/repositories",
    tags=[
        "repositories",
    ],
    responses={404: {"description": "Not found"}},
)


@repositories_router.post("/get_forecast")
async def get_forecast(
    message: githubmonitor.api.schemas.CommitHistorySchema
) -> StreamingResponse:
    """Process the repository historic information and creates a forecast using the last year of available data. After the user logs into the system, he will searches and select a repository. This action triggers this function via the start_session endopoint. repo_commit_history will be transformed into a pd.DataFrame and it will be passed as input for all the pretrainned models (RandomForest, Xgboost, LightGBM), located at the root/models directory.The predictions will be passed again as a json file that will feed the front-end and add the additional weekly obsertvations at the end of the plot.

    Args:
        message (agents.api.schemas.SessionSchema): The request body containing the scan message.
        - repo_id: Id of the repository the user is searching
        - repo_commit_hist: Contains repository last year information.
        - session_id: session id, associated to the user_id. Everytime a user logs into the system a new session will be created

    Returns:
        StreamingResponse: Forcast output passed as a Json object.
    """
    # Make predictions on the testing data
    df_repos = pd.DataFrame()
    dataSeries = message.dataSeries
    print("VALID FORMAT")
    for i, data in enumerate(dataSeries):
        commit_history = data.commits
        dates = data.dates
        logging.info(f"DATES: \n, {dates}")
        if (
            len(dates) <= 60
        ):  # Clusters were trainned with 60 weeks of data to include at least 1 year of information. This information is important to allow the
            raise ValueError(
                "Error: The repository needs at least 60 weeks of historic information to create a forecast"
            )
        # Create a DataFrame
        df = pd.DataFrame({"date": dates, "commit_count": commit_history})
        df["repo_name"] = f"repo_{i}"
        if len(df_repos) ==0:
            df_repos=df
        else:
            df_repos=pd.concat([df, df_repos])
    del df
    print("VALIED DF ENTRIES ")



    # Optionally, convert the 'date' column to datetime format
    df_repos["date"] = pd.to_datetime(df_repos["date"]).dt.normalize()
    # Find the closest Monday for each date
    df_repos["date"] = df_repos["date"] + pd.to_timedelta(
        (6 - df_repos["date"].dt.dayofweek) % 7, unit="D"
    )

    df_repos["date"] = df_repos.date.apply(lambda x: str(x)[:10])
    df_repos["date"] = pd.to_datetime(df_repos.date)

    df_out = df_repos[["date", "commit_count", "repo_name"]]
    n_lag_plot = 7  # Showing last 7 weeks
    df_out = df_out[df_out.date >= df_out.date.max() - timedelta(days=7 * n_lag_plot)]
    df_out = df_out.rename(
        columns={"commit_count": "response_forecast", "date": "response_date", "repo_name":"repo_name"}
    )

    # Feature Engineering processs
    lag_list = [2, 4, 6, 10]
    rolling_list = [2, 4, 6]

    df_window_mean, df_window_ewm = moving_average_variables(
        df_repos, "date", lag_list, rolling_list
    )

    df_features = df_repos.merge(df_window_mean, on=["date", "repo_name"], how="inner")
    df_features = df_features.merge(
        df_window_ewm, on=["date", "repo_name"], how="inner"
    )
    df_features = df_features.sort_values(["repo_name", "date"], ascending=True)

    # ITERATIVE PREDICTION
    evaluation_window = max(lag_list) + max(rolling_list) + 1
    prediction_window = 12
    forecast_start = df_features["date"].max()
    min_date = forecast_start - timedelta(days=(evaluation_window * 7))
    parallel = True

    # Order is important and avoid making mistakes on the prediction
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

    for model in ["xgboost", "randomforest", "lgbm"]:
        start_time = time.time()
        df_iter = create_iterative_forecast(
            df_features,
            model=model,
            forecast_start=forecast_start,
            lag_list=lag_list,
            rolling_list=rolling_list,
            parallel=parallel,
            evaluation_window=evaluation_window,
            min_date=min_date,
            production=True,
            prediction_window=prediction_window,
            model_features=model_features,
        )

        end_time = time.time()
        elapsed_time = round(end_time - start_time)
        logging.info(f" MODEL: {model} - TRAINNING TIME: {elapsed_time}")

    # ENSEMBLE PREDICTION
    retrain_models = False

    # Start from the last day and starts making iterations over the future
    prediction_start = forecast_start
    min_date = prediction_start - timedelta(days=(evaluation_window * 7))

    cut_date = df_iter.Date.max()
    # df_forecast is loaded from the begining
    df_predicted_s1, df_models = prepare_all_models(
        retrain_models, cut_date=cut_date, production=True
    )
    df_forecast_month = create_month_dummy(
        df_models, retrain_models=retrain_models, production=True
    )
    df_forecast_horizon = create_forecast_horizon(df_models.copy())
    df_target = df_predicted_s1[["Repository", "Date", "Commit Real"]].copy()
    df_target = (
        df_target.groupby(["Repository", "Date"])["Commit Real"].first().reset_index()
    )
    predicted_labels = set_cluster(
        df=df_features,
        date_col="date",
        index_cols=["repo_name"],
        target="commit_count",
    )  # Note that we are using the initial DF with all the information

    n_cluster = 8
    df_clusters = df_predicted_s1.copy()
    cluster_cols = []
    for i in range(n_cluster):
        col = f"Cluster_{i}"
        df_clusters[col] = 0  # Complete clusters accordingly
        cluster_cols.append(col)
    cluster = predicted_labels[0]

    df_clusters[f"Cluster_{cluster}"] = 1
    df_clusters["cluster"] = cluster
    cluster_cols.append("cluster")
    df_clusters = df_clusters.groupby(["Repository", "Date"])[cluster_cols].first()

    df_lr = df_models.merge(df_clusters, on=["Repository", "Date"], how="left")
    test_uniqueness_branch_date(df_models)
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

    model, df_results = train_elasticnet_ensemble_model(
        df_model_input,
        target="Commit Real",
        retrain_models=retrain_models,
        load_model=True,
        production=True,
    )

    df_results["model_family"] = "elasticnet"
    df_results = df_results[["elasticnet", "Date", "Repository"]]
    df_results = df_results.rename(
        columns={"elasticnet": "response_forecast", "Date": "response_date", "Repository":"repo_name"}
    )

    df_out = pd.concat([df_out, df_results])
    forecast = []
    for repo in sorted(df_out["repo_name"].unique()):
        df_repo = df_out[df_out.repo_name == repo]
        response = {
            "dates": df_repo["response_date"].values.tolist(),
            "forecast": df_repo["response_forecast"].values.tolist(),
        }
        forecast.append(response)
        
        
    return JSONResponse(forecast)


async def create_forecast(
    session_id: str, repo_id: str, repo_commit_hist: object
) -> StreamingResponse:
    """Process the repository historic information and creates a forecast using the last year of available data. After the user logs into the system, he will searches and select a repository. This action triggers this function via the start_session endopoint. repo_commit_history will be transformed into a pd.DataFrame and it will be passed as input for all the pretrainned models (RandomForest, Xgboost, LightGBM), located at the root/models directory.The predictions will be passed again as a json file that will feed the front-end and add the additional weekly obsertvations at the end of the plot.

    Args:
        message (agents.api.schemas.SessionSchema): The request body containing the scan message.
        - repo_id: Id of the repository the user is searching
        - repo_commit_hist: Contains repository last year information.
        - session_id: session id, associated to the user_id. Everytime a user logs into the system a new session will be created

    Returns:
        StreamingResponse: Forcast output passed as a Json object.
    """
    # Create placehodler to test connection
    logging.warning("Crete forecast is not connected")
    return "FORECAST FUNCTION"
