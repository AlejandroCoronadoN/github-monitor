"""Common utilities for langchain."""
import asyncio
import random
import re
import time
from typing import Any, Generator

from config.configuration import SETTINGS
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI


def stream_static_text(
    text: str, speed_multiplier: float = 0
) -> Generator[str, Any, None]:
    """Yields a generator for static text."""
    gearing = 1.0 - speed_multiplier
    for chunk in re.split(r"(\(|\))", text):
        if re.match(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            chunk,
        ):
            yield chunk
        else:
            for token in chunk:
                yield token
                time.sleep(
                    random.randrange(
                        SETTINGS.minimum_delay_ms * gearing, SETTINGS.maximum_delay_ms
                    )
                    / 1_000
                )


class StreamingCallbackHandler(BaseCallbackHandler):
    """A class that handles streaming callbacks.

    This class inherits from the BaseCallbackHandler class and provides methods for handling streaming callbacks.

    Attributes:
        None
    """

    def __init__(self):
        """Inits it."""
        self.queue = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """This function is called when a new token is generated and adds the token to the queue.

        Args:
            token (str): The new token generated.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        await self.queue.put(token)

    async def clear_queue(self) -> None:
        """Clears all items from the queue.

        This method will remove all items from the queue until it is empty.

        Returns:
            None
        """
        while not self.queue.empty():
            await asyncio.wait_for(self.queue.get(), timeout=1)


streaming_callback = StreamingCallbackHandler()

llm = ChatOpenAI(
    openai_api_key=SETTINGS.openai_api_key,
    temperature=0.2,
    model_name="davinci",
    streaming=True,
    callbacks=[streaming_callback],
)
