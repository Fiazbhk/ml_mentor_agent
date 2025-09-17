#!/bin/bash
cd "$(dirname "$0")"
7
if [ -d ".venv" ]; then
source .venv/bin/activate
fi
uv run python main.py
