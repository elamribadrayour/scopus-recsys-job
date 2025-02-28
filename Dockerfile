FROM python:3.12.4-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN mkdir -p /app /data

WORKDIR /app

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen

COPY src/job /app/src/job
WORKDIR /app/src/job

CMD ["uv", "run", "main.py", "init"]
