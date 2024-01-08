"""This module focuses on user-related information. For the first version of the application we will only log the user ID to test that the FastAPI integration is working. This module can be used to store and retrieve user information inside the Users DB and link user_id with the session DataBase. That way we can recover user information everytime he login, pass the information of the repositories he wants to compare and retrieve the forecast of each repository, avoiding recalcultaing it everytime."""

import logging

import githubmonitor.api.schemas
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from loguru import logger

logging.basicConfig(level=logging.INFO)

# Router basic config
users_router = APIRouter(
    prefix="/users",
    tags=[
        "users",
    ],
    responses={404: {"description": "User not found"}},
)


@users_router.post("/scan_user")
async def scan_user(
    message: githubmonitor.api.schemas.UserSchema,
) -> StreamingResponse:
    """Send a message to the client using Server-Sent Events (SSE).

    Args:
        message (agents.api.schemas.UserSchema): The request body containing the user scan message.

    Returns:
        StreamingResponse: A streaming response object with the user scan details.
    """
    user_id = message.user_id
    logger.info(f"User message: {user_id}")
    user_message = {"user_id": user_id}
    return user_message
