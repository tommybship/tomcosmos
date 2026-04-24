import pytest

from tomcosmos import BODY_CONSTANTS, UnknownBodyError, resolve_body_constant


def test_all_expected_bodies_present() -> None:
    expected = {"sun", "mercury", "venus", "earth", "moon",
                "mars", "jupiter", "saturn", "uranus", "neptune"}
    assert expected <= set(BODY_CONSTANTS)


def test_resolve_by_canonical_name() -> None:
    b = resolve_body_constant("earth")
    assert b.name == "earth"
    assert b.spice_id == 399
    assert b.mass_kg == pytest.approx(5.9724e24, rel=1e-6)


def test_resolve_case_insensitive() -> None:
    assert resolve_body_constant("EARTH") is resolve_body_constant("earth")
    assert resolve_body_constant("  Mars ") is resolve_body_constant("mars")


def test_resolve_by_spice_id() -> None:
    assert resolve_body_constant(399).name == "earth"
    assert resolve_body_constant(10).name == "sun"


def test_resolve_by_alias() -> None:
    assert resolve_body_constant("sol") is resolve_body_constant("sun")
    assert resolve_body_constant("terra") is resolve_body_constant("earth")
    assert resolve_body_constant("luna") is resolve_body_constant("moon")


def test_unknown_name_raises_with_suggestion() -> None:
    with pytest.raises(UnknownBodyError, match="earth"):
        resolve_body_constant("erth")


def test_unknown_name_without_close_match() -> None:
    with pytest.raises(UnknownBodyError):
        resolve_body_constant("xyzzy")


def test_unknown_spice_id_raises() -> None:
    with pytest.raises(UnknownBodyError, match="12345"):
        resolve_body_constant(12345)


def test_bad_type_raises() -> None:
    with pytest.raises(TypeError):
        resolve_body_constant(3.14)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        resolve_body_constant(True)  # type: ignore[arg-type]


def test_colors_are_valid_hex() -> None:
    import re
    pat = re.compile(r"^#[0-9A-Fa-f]{6}$")
    for body in BODY_CONSTANTS.values():
        assert pat.match(body.color_hex), f"{body.name}: {body.color_hex!r}"


def test_spice_ids_are_unique() -> None:
    ids = [b.spice_id for b in BODY_CONSTANTS.values()]
    assert len(ids) == len(set(ids))
