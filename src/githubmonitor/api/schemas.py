"""Schemas for the API."""

from typing import List, Any

from pydantic import BaseModel, Field


## API Schemas
class UserSchema(BaseModel):
    """Represents the id passed by the scanner when a iser credential is scanned.

    Args:
        message (str): The content of the message.
    """

    user_id: int = Field(..., example=10001)


class MessageSchema(BaseModel):
    """Represents the id passed by the scanner when a iser credential is scanned.

    Args:
        message (str): The content of the message.
    """

    query: str = Field(
        ...,
        example="*** A project that implements React and python to create powerful insights about git repositories. This project implements langchain and ML models",
    )


class RepositorySchema(BaseModel):
    """Represents the id passed by the scanner when a iser credential is scanned.

    Args:
        message (str): The content of the message.
    """

    product_id: int = Field(..., example=92467)
    user_id: int = Field(..., example="10001")

class CommitHistoryItem(BaseModel):
    dates: List[str]
    commits: List[int]


class CommitHistorySchema(BaseModel):
    dataSeries: List[CommitHistoryItem] = Field(
        ...,
        example=[
            {"dates": ["2023-03-26T23:27:42.132Z", "2023-03-19T23:27:42.132Z", "2023-03-12T23:27:42.132Z"],
             "commits": [1, 2, 3]},
            {"dates": ["2023-03-26T23:27:42.132Z", "2023-03-19T23:27:42.132Z", "2023-03-12T23:27:42.132Z"],
             "commits": [1, 2, 3]}
        ]
    )