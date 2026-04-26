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


def cache_dir() -> Path:
    """Where deterministic external-API responses get cached so we
    don't re-query JPL endpoints on every test run / scenario assembly.
    Currently used by `tomcosmos.targeting.horizons` for state-vector
    queries, which are deterministic in (designation, epoch_jd) and
    cost ~1.3 s round-trip each.

    Env: `TOMCOSMOS_CACHE_DIR`. Default: `./data/cache` relative to cwd.
    Safe to delete at any time — every entry is re-fetchable.
    """
    return Path(os.environ.get("TOMCOSMOS_CACHE_DIR", "data/cache"))


def log_level() -> str:
    """Default log level for `run` when `--verbose` isn't set.

    Env: `TOMCOSMOS_LOG_LEVEL`. Default: `info`. CLI `--verbose` forces `debug`.
    """
    return os.environ.get("TOMCOSMOS_LOG_LEVEL", "info").lower()


def assist_planet_kernel() -> Path:
    """Path to ASSIST's planet ephemeris (DE440 / DE441 binary SPK).

    Env: `TOMCOSMOS_ASSIST_PLANET_KERNEL`. Default:
    `<kernel_dir>/de440.bsp`. ASSIST also accepts `de441.bsp` or the
    JPL ASCII ephemerides (`linux_p1550p2650.440`,
    `linux_m13000p17000.441`); set the env var to point at whichever
    you have downloaded. A scenario with `integrator.ephemeris_perturbers
    = true` requires this file to be present.
    """
    override = os.environ.get("TOMCOSMOS_ASSIST_PLANET_KERNEL")
    if override:
        return Path(override)
    return kernel_dir() / "de440.bsp"


def assist_asteroid_kernel() -> Path:
    """Path to ASSIST's asteroid-perturber kernel (sb441-n16.bsp).

    Env: `TOMCOSMOS_ASSIST_ASTEROID_KERNEL`. Default:
    `<kernel_dir>/sb441-n16.bsp`. Required when
    `integrator.ephemeris_perturbers = true`.
    """
    override = os.environ.get("TOMCOSMOS_ASSIST_ASTEROID_KERNEL")
    if override:
        return Path(override)
    return kernel_dir() / "sb441-n16.bsp"
