"""Clasify issues and other github information."""


async def llm_sentiment(description: str) -> str:
    """This Tool make a 280 characters (maximun) description using the repository description as base.

    Args:
        user_id (str): User's Id

    Returns:
        str:  Formatted message with all the near branches information.
    """
    return description
