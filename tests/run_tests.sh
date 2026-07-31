#!/usr/bin/env bash
# sas-patch-service test suite (pytest). Needs the repo venv; the surgepy-backed
# tests self-skip when the native module isn't built.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m pytest tests/ -q "$@"
