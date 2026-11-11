#!/usr/bin/env python3
"""Oracle entrypoint — runs the full pipeline end-to-end.

This is a thin CLI wrapper. All real logic lives in src/pipeline.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src directory is in path if executed from root
CURRENT_DIR = Path(__file__).resolve().parent
if (CURRENT_DIR / "src").exists():
    sys.path.append(str(CURRENT_DIR / "src"))

try:
    from pipeline import run_full
except ImportError:
    from src.pipeline import run_full

if __name__ == "__main__":
    run_full()