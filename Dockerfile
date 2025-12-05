FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY . .

RUN uv pip install --system --no-cache -e .

ENTRYPOINT ["python", "scripts/run_agent.py"]
CMD ["--help"]

