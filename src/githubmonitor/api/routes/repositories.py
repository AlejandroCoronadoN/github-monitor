"""Repositorie routes.

This module contains the forecast process of the application. by passing the historic information of a Github repository, this module uses create_forecast function to call pretrainned models and functions that live in the features and models directory of this project. These functions have been used previously to test and tune our models. This is the production version of the forecast algorithm and We are only making predictions and testing them trough the github_forecast endpoint.

"""

import logging

import githubmonitor.api.schemas
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
# Router basic config
scan_router = APIRouter(
    prefix="/repositories",
    tags=[
        "repositories",
    ],
    responses={404: {"description": "Not found"}},
)


@scan_router.post("/get_forecast")
async def scan_product(
    message: githubmonitor.api.schemas.RepositorySchema
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
    repo_id = message.repo_id
    session_id = message.session_id
    repo_commit_hist = message.repo_commit_hist
    forecast = create_forecast(session_id, repo_id, repo_commit_hist)

    response = {"message": "A NEW FORECAST WAS CREATED", "forecast": forecast}

    return response


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
