"""Tests for optional physics effects (GR today, SRP / Yarkovsky later).

Most tests use a minimal Sun-Mercury two-body scenario so Newtonian
dynamics produce zero precession and any shift under GR is unambiguous.
"""
from __future__ import annotations

import numpy as np
import pytest

from tomcosmos import Scenario, run
from tomcosmos.state.ephemeris import SkyfieldSource

AU_KM = 1.495978707e8


def _sun_mercury_scenario(
    *, duration: str, with_gr: bool, name: str = "test"
) -> Scenario:
    return Scenario.model_validate(
        {
            "schema_version": 1,
            "name": name,
            "epoch": "2000-01-01T12:00:00 TDB",
            "duration": duration,
            "integrator": {
                "name": "whfast", "timestep": "0.1 day",
                "effects": ["gr"] if with_gr else [],
            },
            "output": {"format": "parquet", "cadence": "1 yr"},
            "bodies": [
                {"name": "sun", "mass_kg": 1.989e30, "radius_km": 695700.0,
                 "ic": {"source": "explicit",
                        "r": [0, 0, 0], "v": [0, 0, 0]}},
                {"name": "mercury", "mass_kg": 3.3011e23, "radius_km": 2439.7,
                 "ic": {"source": "explicit",
                        "r": [0.3075 * AU_KM, 0, 0],
                        "v": [0, 58.98, 0]}},
            ],
        }
    )


class _NoEphemerisNeeded(SkyfieldSource):  # type: ignore[misc]
    def __init__(self) -> None: pass
    def query(self, body, epoch): raise AssertionError  # type: ignore[no-untyped-def]
    def available_bodies(self): return ()  # type: ignore[override]
    def time_range(self):  # type: ignore[override]
        from astropy.time import Time
        return Time("1900-01-01", scale="tdb"), Time("2200-01-01", scale="tdb")


# --- Schema validation -------------------------------------------------------


def test_effects_defaults_to_empty() -> None:
    s = _sun_mercury_scenario(duration="1 yr", with_gr=False)
    assert s.integrator.effects == []


def test_effects_accepts_gr() -> None:
    s = _sun_mercury_scenario(duration="1 yr", with_gr=True)
    assert s.integrator.effects == ["gr"]


def test_effects_rejects_unknown() -> None:
    with pytest.raises(Exception, match="gr|Literal|input"):
        Scenario.model_validate(
            {
                "schema_version": 1, "name": "bad",
                "epoch": "2026-01-01T00:00:00 TDB", "duration": "1 yr",
                "integrator": {
                    "name": "whfast", "timestep": "1 day",
                    "effects": ["yarkovsky"],
                },
                "output": {"format": "parquet", "cadence": "1 day"},
                "bodies": [
                    {"name": "sun", "mass_kg": 1.989e30, "radius_km": 695700.0,
                     "ic": {"source": "explicit",
                            "r": [0, 0, 0], "v": [0, 0, 0]}},
                ],
            }
        )


def test_effects_rejects_duplicates() -> None:
    with pytest.raises(Exception, match="duplicate"):
        Scenario.model_validate(
            {
                "schema_version": 1, "name": "bad",
                "epoch": "2026-01-01T00:00:00 TDB", "duration": "1 yr",
                "integrator": {
                    "name": "whfast", "timestep": "1 day",
                    "effects": ["gr", "gr"],
                },
                "output": {"format": "parquet", "cadence": "1 day"},
                "bodies": [
                    {"name": "sun", "mass_kg": 1.989e30, "radius_km": 695700.0,
                     "ic": {"source": "explicit",
                            "r": [0, 0, 0], "v": [0, 0, 0]}},
                ],
            }
        )


def test_gr_requires_sun_body() -> None:
    with pytest.raises(Exception, match="gr.*requires a body named 'sun'"):
        Scenario.model_validate(
            {
                "schema_version": 1, "name": "bad",
                "epoch": "2026-01-01T00:00:00 TDB", "duration": "1 yr",
                "integrator": {
                    "name": "whfast", "timestep": "1 day",
                    "effects": ["gr"],
                },
                "output": {"format": "parquet", "cadence": "1 day"},
                "bodies": [
                    {"name": "alpha", "mass_kg": 1.989e30, "radius_km": 695700.0,
                     "ic": {"source": "explicit",
                            "r": [0, 0, 0], "v": [0, 0, 0]}},
                ],
            }
        )


# --- Runtime behavior --------------------------------------------------------


def test_gr_sets_velocity_dependent_flag() -> None:
    """WHFast needs this flag or it treats the force as instantaneous."""
    from tomcosmos.state.ic import resolve_scenario
    from tomcosmos.state.integrator import build_simulation

    s = _sun_mercury_scenario(duration="1 yr", with_gr=True)
    bodies, particles = resolve_scenario(s, _NoEphemerisNeeded())
    sim = build_simulation(bodies, particles, s.integrator)
    assert sim.force_is_velocity_dependent == 1
    # Callback kept alive on the sim so Python doesn't GC it
    assert hasattr(sim, "_tomcosmos_effect_callbacks")


def test_no_effects_no_velocity_dependent_flag() -> None:
    from tomcosmos.state.ic import resolve_scenario
    from tomcosmos.state.integrator import build_simulation

    s = _sun_mercury_scenario(duration="1 yr", with_gr=False)
    bodies, particles = resolve_scenario(s, _NoEphemerisNeeded())
    sim = build_simulation(bodies, particles, s.integrator)
    assert sim.force_is_velocity_dependent == 0
    assert not hasattr(sim, "_tomcosmos_effect_callbacks")


def test_gr_and_newton_positions_start_identical() -> None:
    """At t=0, GR hasn't had time to shift anything."""
    h_n = run(_sun_mercury_scenario(duration="1 yr", with_gr=False),
              source=_NoEphemerisNeeded())
    h_g = run(_sun_mercury_scenario(duration="1 yr", with_gr=True),
              source=_NoEphemerisNeeded())
    m_n = h_n.body_trajectory("mercury")
    m_g = h_g.body_trajectory("mercury")
    r_n_0 = m_n.iloc[0][["x", "y", "z"]].to_numpy()
    r_g_0 = m_g.iloc[0][["x", "y", "z"]].to_numpy()
    assert np.allclose(r_n_0, r_g_0)


def test_gr_shifts_mercury_position_over_time() -> None:
    """Mercury's 1PN perihelion precession is ~43 arcsec/century; over
    10 years that's a few arcsec, translating to thousands of km of
    position shift versus pure Newtonian at Mercury's 0.4 AU orbit."""
    h_n = run(_sun_mercury_scenario(duration="10 yr", with_gr=False),
              source=_NoEphemerisNeeded())
    h_g = run(_sun_mercury_scenario(duration="10 yr", with_gr=True),
              source=_NoEphemerisNeeded())
    m_n = h_n.body_trajectory("mercury")
    m_g = h_g.body_trajectory("mercury")
    r_n_end = m_n.iloc[-1][["x", "y", "z"]].to_numpy()
    r_g_end = m_g.iloc[-1][["x", "y", "z"]].to_numpy()
    delta_km = float(np.linalg.norm(r_n_end - r_g_end))
    # Expect ~1000-5000 km over 10 yr; wide band to avoid flakes.
    assert 500 < delta_km < 20_000, f"GR vs Newton shift at 10 yr: {delta_km:.0f} km"


def test_gr_signature_scales_with_time() -> None:
    """Position shift should grow roughly linearly (with oscillations) as
    precession accumulates."""
    h_n = run(_sun_mercury_scenario(duration="20 yr", with_gr=False),
              source=_NoEphemerisNeeded())
    h_g = run(_sun_mercury_scenario(duration="20 yr", with_gr=True),
              source=_NoEphemerisNeeded())
    m_n = h_n.body_trajectory("mercury")
    m_g = h_g.body_trajectory("mercury")
    deltas_km = []
    # sample at 5, 10, 20 yr
    for t_target in (5, 10, 20):
        i = min(t_target, len(m_n) - 1)
        r_n = m_n.iloc[i][["x", "y", "z"]].to_numpy()
        r_g = m_g.iloc[i][["x", "y", "z"]].to_numpy()
        deltas_km.append(float(np.linalg.norm(r_n - r_g)))
    # Monotone-ish growth (allow ~30% wiggle for orbital-phase effects)
    assert deltas_km[2] > deltas_km[0], (
        f"GR shift should grow over time; got {deltas_km}"
    )


def test_gr_energy_error_bounded_but_larger_than_pure_newtonian() -> None:
    """With GR enabled, WHFast loses strict symplecticity (force is
    velocity-dependent). Energy still bounded, just not at 1e-13 anymore."""
    h_n = run(_sun_mercury_scenario(duration="10 yr", with_gr=False),
              source=_NoEphemerisNeeded())
    h_g = run(_sun_mercury_scenario(duration="10 yr", with_gr=True),
              source=_NoEphemerisNeeded())
    max_err_n = float(h_n.df["energy_rel_err"].max())
    max_err_g = float(h_g.df["energy_rel_err"].max())
    assert max_err_n < 1e-10          # pure Newton: machine-precision bounded
    assert max_err_g < 1e-4           # GR: loose bound, still physical
    assert max_err_g > max_err_n      # GR really is less tight
