"""Session routes.

This module defines the logic that connects each user to his own user session. In the Session DB we will only have three variables, user_id, session_id, repo_id. Everytime a user login his session will be reseted, and as long he is logged all his repositories searches will be passed to the front end. repo_id will be used to stored the data asociated to the repository search in it's own DataBase Reposotories DB.

"""


import logging

import githubmonitor.api.schemas
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from githubmonitor.api.routes.repositories import create_forecast

logging.basicConfig(level=logging.INFO)
# Router basic config
sessions_router = APIRouter(
    prefix="/sessions",
    tags=[
        "sessions",
    ],
    responses={404: {"description": "Not found"}},
)


@sessions_router.post("/start_session")
async def scan_repo(
    message: githubmonitor.api.schemas.RepositorySchema
) -> StreamingResponse:
    """Represents a repository being searched by the user. This function creates a new entry in the sessions DB that connects each user with a different session. Then we use the session_id and the repository information to store the data in the repositories DB. After saving the new entry the repository information is used to create a new forecast for the following 3 months.

    Args:
        message (agents.api.schemas.SessionSchema): The request body containing the scan message.
        - message.repo_id: Id of the repository the user is searching
        - message.repo_commit_hist: Contains repository last year information.
        - message.user_id: User ID (default to 1000)

    Returns:
        StreamingResponse: A streaming response object with the scan details.
    """
    repo_id = message.repo_id
    user_id = message.user_id
    session_id = message.session_id
    repo_commit_hist = message.repo_commit_hist
    logging.info(f"{repo_id} - {user_id} - {session_id} - {repo_commit_hist}")
    # forecast contains the same
    forecast = create_forecast(session_id, repo_id, repo_commit_hist)

    response = {"message": "A NEW SESSION WAS CREATED", "forecast": forecast}

    return response
