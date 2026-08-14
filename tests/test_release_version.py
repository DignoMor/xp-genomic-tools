"""Release metadata contract for the 0.1.0a2 code cut."""

from importlib.metadata import version


def test_package_version_matches_release() -> None:
    """The installed distribution reports the release version being cut."""
    assert version("RGTools") == "0.1.0a2"
