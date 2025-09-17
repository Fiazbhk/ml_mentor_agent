FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"
COPY pyproject.toml ./
COPY requirements.txt ./
COPY main.py ./
COPY ml_solver.py ./
COPY run_agent.sh ./
RUN uv venv .venv && . .venv/bin/activate && uv pip install -r requirements.txt
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "main.py"]
