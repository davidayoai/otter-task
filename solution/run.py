#!/usr/bin/env python3
"""Oracle entrypoint — runs the full pipeline end-to-end.

This is a thin CLI wrapper. All real logic lives in src/pipeline.py.
"""
from pipeline import run_full

if __name__ == "__main__":
    run_full()
