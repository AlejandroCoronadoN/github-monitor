"""Schemas for the API."""

from pydantic import BaseModel, Field


## API Schemas
class UserSchema(BaseModel):
    """Represents the id passed by the scanner when a iser credential is scanned.

    Args:
        message (str): The content of the message.
    """

    user_id: int = Field(..., example=10001)


class RepositorySchema(BaseModel):
    """Represents the id passed by the scanner when a iser credential is scanned.

    Args:
        message (str): The content of the message.
    """

    product_id: int = Field(..., example=92467)
    user_id: int = Field(..., example="10001")
