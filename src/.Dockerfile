FROM python:3.11-buster

RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.11 python3.11-dev && curl -sSL https://install.python-poetry.org | python3.11 - && echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc && . ~/.bashrc && cd /app && poetry install && uvicorn githubmonitor.main:app --reload --host 0.0.0.0

