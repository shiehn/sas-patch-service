"""sas-patch-service core library (Phase 0 spike)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SURGE_SRC = REPO_ROOT / "third_party" / "surge"
SAS_APP = REPO_ROOT.parent / "sas-app"
