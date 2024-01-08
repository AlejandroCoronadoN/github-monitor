"""Summarizes issues and other github information."""


async def nlp_description(description: str) -> str:
    """This Tool make a 280 characters (maximun) description using the repository description as base.

    Args:
        user_id (str): User's Id

    Returns:
        str:  Formatted message with all the near branches information.
    """
    return description
