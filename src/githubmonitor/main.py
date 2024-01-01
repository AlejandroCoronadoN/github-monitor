"""Main Entrypoint."""

from config.configuration import SETTINGS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from githubmonitor.api.routes.scans import scan_router
from githubmonitor.api.routes.transactions import transaction_router
from loguru import logger

app = FastAPI(
    title="Github Monitor",
    description="An application that helps users identify the repositories that will get the most support over the following 3 months.",
    version=SETTINGS.version,
)
app.include_router(router=transaction_router)
app.include_router(router=scan_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="../frontend/build/static"), name="static")
app.mount("/assets", StaticFiles(directory="../frontend/build/assets"), name="static")


@app.get("/")
async def read_index() -> HTMLResponse:
    """This function reads the contents of the index.html file and returns it as an HTMLResponse.

    :return: The contents of the index.html file as an HTMLResponse.
    :rtype: HTMLResponse
    """
    with open("../frontend/build/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/ping")
async def ping():
    """A function that returns a dictionary with a message key and a value of "pong 🏓"."""
    logger.info("Ping received")
    return {"message": "pong 🏓"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
