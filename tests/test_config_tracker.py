"""Tests for TrackerConfig class."""

import pytest

from nipoppy.config.tracker import TrackerConfig

FIELDS_STEP = [
    "PATHS",
    "PARTICIPANT_SESSION_DIR",
]


@pytest.mark.parametrize(
    "data",
    [
        {"PATHS": ["path1", "path2"]},
    ],
)
def test_fields(data):
    tracker_config = TrackerConfig(**data)
    for field in FIELDS_STEP:
        assert hasattr(tracker_config, field)

    assert len(set(tracker_config.model_dump())) == len(FIELDS_STEP)


def test_no_extra_field():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        TrackerConfig(not_a_field="a")


def test_at_least_one_path():
    with pytest.raises(ValueError, match="must contain at least one path"):
        TrackerConfig(PATHS=[])
