"""Runtime configuration — filesystem paths and log level.

Resolution order for every value: CLI flag > env var > default. The CLI
layer (cli.py) is responsible for overriding env-derived defaults with
flag values; library callers can either let env vars drive things or
pass explicit paths.

See PLAN.md > "Environment variables" for the pinned contract.
"""
from __future__ import annotations

import os
from pathlib import Path


def kernel_dir() -> Path:
    """Where `fetch-kernels` writes and ephemeris sources read.

    Env: `TOMCOSMOS_KERNEL_DIR`. Default: `./data/kernels` relative to cwd.
    """
    return Path(os.environ.get("TOMCOSMOS_KERNEL_DIR", "data/kernels"))


def runs_dir() -> Path:
    """Where `run` writes Parquet + logs when `output.path` is relative.

    Env: `TOMCOSMOS_RUNS_DIR`. Default: `./runs` relative to cwd.
    """
    return Path(os.environ.get("TOMCOSMOS_RUNS_DIR", "runs"))


def log_level() -> str:
    """Default log level for `run` when `--verbose` isn't set.

    Env: `TOMCOSMOS_LOG_LEVEL`. Default: `info`. CLI `--verbose` forces `debug`.
    """
    return os.environ.get("TOMCOSMOS_LOG_LEVEL", "info").lower()
