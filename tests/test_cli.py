"""Tests for the CLI."""

from pathlib import Path

import pytest

from nipoppy.cli.run import cli

from .conftest import ATTR_TO_DPATH_MAP


def test_cli():
    with pytest.raises(SystemExit) as exception:
        cli(["nipoppy", "-h"])
    assert exception.value.code == 0


def test_cli_invalid():
    with pytest.raises(SystemExit) as exception:
        cli(["nipoppy", "--fake-arg"])
    assert exception.value.code != 0


def test_cli_init(tmp_path: Path):
    try:
        cli(["nipoppy", "init", str(tmp_path / "my_dataset")]) is None
    except SystemExit:
        pass


def test_cli_status(tmp_path: Path):
    try:
        cli(["nipoppy", "status", str(tmp_path / "my_dataset")]) is None
    except SystemExit:
        pass


def test_cli_doughnut(tmp_path: Path):
    dpath_root = tmp_path / "my_dataset"
    try:
        cli(["nipoppy", "doughnut", str(dpath_root)])
    except BaseException:
        pass

    # check that a logfile was created
    assert (
        len(list((dpath_root / ATTR_TO_DPATH_MAP["dpath_logs"]).glob("doughnut/*.log")))
        == 1
    )


def test_cli_reorg(tmp_path: Path):
    dpath_root = tmp_path / "my_dataset"
    try:
        cli(["nipoppy", "reorg", str(dpath_root)])
    except BaseException:
        pass

    # check that a logfile was created
    assert (
        len(
            list(
                (dpath_root / ATTR_TO_DPATH_MAP["dpath_logs"]).glob("dicom_reorg/*.log")
            )
        )
        == 1
    )


def test_cli_bidsify(tmp_path: Path):
    dpath_root = tmp_path / "my_dataset"
    try:
        cli(
            [
                "nipoppy",
                "bidsify",
                str(dpath_root),
                "--pipeline",
                "my_pipeline",
                "--pipeline-version",
                "1.0",
                "--pipeline-step",
                "step1",
            ]
        )
    except BaseException:
        pass

    # check that a logfile was created
    assert (
        len(
            list(
                (dpath_root / ATTR_TO_DPATH_MAP["dpath_logs"]).glob(
                    "bids_conversion/my_pipeline-1.0/*.log"
                )
            )
        )
        == 1
    )


def test_cli_run(tmp_path: Path):
    dpath_root = tmp_path / "my_dataset"
    try:
        cli(
            [
                "nipoppy",
                "run",
                str(dpath_root),
                "--pipeline",
                "my_pipeline",
                "--pipeline-version",
                "1.0",
            ]
        )
    except BaseException:
        pass

    # check that a logfile was created
    assert (
        len(
            list(
                (dpath_root / ATTR_TO_DPATH_MAP["dpath_logs"]).glob(
                    "run/my_pipeline-1.0/*.log"
                )
            )
        )
        == 1
    )


def test_cli_track(tmp_path: Path):
    dpath_root = tmp_path / "my_dataset"
    try:
        cli(
            [
                "nipoppy",
                "track",
                str(dpath_root),
                "--pipeline",
                "my_pipeline",
                "--pipeline-version",
                "1.0",
            ]
        )
    except BaseException:
        pass

    # check that a logfile was created
    assert (
        len(
            list(
                (dpath_root / ATTR_TO_DPATH_MAP["dpath_logs"]).glob(
                    "track/my_pipeline-1.0/*.log"
                )
            )
        )
        == 1
    )


def test_cli_extract(tmp_path: Path):
    dpath_root = tmp_path / "my_dataset"
    try:
        cli(
            [
                "nipoppy",
                "extract",
                str(dpath_root),
                "--pipeline",
                "my_pipeline",
                "--pipeline-version",
                "1.0",
            ]
        )
    except BaseException:
        pass

    # check that a logfile was created
    assert (
        len(
            list(
                (dpath_root / ATTR_TO_DPATH_MAP["dpath_logs"]).glob(
                    "extract/my_pipeline-1.0/*.log"
                )
            )
        )
        == 1
    )
