#!/usr/bin/env bash
# sas-patch-service test suite (pytest). Needs the repo venv; the surgepy-backed
# tests self-skip when the native module isn't built.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x ".venv/bin/python" ]]; then
  exec .venv/bin/python -m pytest tests/ -q "$@"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  # Windows venv layout (Git Bash).
  exec .venv/Scripts/python.exe -m pytest tests/ -q "$@"
else
  echo "ERROR: no venv found (.venv/bin/python or .venv/Scripts/python.exe) — create it per README" >&2
  exit 1
fi
