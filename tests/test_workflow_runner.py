"""Tests for PipelineRunner."""

import json
from pathlib import Path

import pytest
import pytest_mock
from bids import BIDSLayout
from fids import fids

from nipoppy.config.main import Config
from nipoppy.tabular.bagel import Bagel
from nipoppy.tabular.doughnut import Doughnut
from nipoppy.workflows.runner import PipelineRunner

from .conftest import create_empty_dataset, get_config, prepare_dataset


@pytest.fixture(scope="function")
def config(tmp_path: Path):
    fpath_descriptor = tmp_path / "descriptor.json"
    fpath_invocation = tmp_path / "invocation.json"

    descriptor = {
        "name": "dummy_pipeline",
        "tool-version": "1.0.0",
        "description": "A dummy pipeline for testing",
        "schema-version": "0.5",
        "command-line": "echo [ARG1] [ARG2] [[NIPOPPY_DPATH_BIDS]]",
        "inputs": [
            {
                "id": "arg1",
                "name": "arg1",
                "type": "String",
                "command-line-flag": "--arg1",
                "value-key": "[ARG1]",
            },
            {
                "id": "arg2",
                "name": "arg2",
                "type": "Number",
                "command-line-flag": "--arg2",
                "value-key": "[ARG2]",
            },
        ],
        "custom": {"nipoppy": {"CONTAINER_SUBCOMMAND": "exec"}},
    }
    invocation = {
        "arg1": "[[NIPOPPY_PARTICIPANT_ID]] [[NIPOPPY_BIDS_SESSION_ID]]",
        "arg2": 10,
    }

    fpath_descriptor.write_text(json.dumps(descriptor))
    fpath_invocation.write_text(json.dumps(invocation))

    return get_config(
        visit_ids=["BL", "V04"],
        container_config={"COMMAND": "echo"},  # dummy command
        proc_pipelines=[
            {
                "NAME": "dummy_pipeline",
                "VERSION": "1.0.0",
                "STEPS": [
                    {
                        "DESCRIPTOR_FILE": fpath_descriptor,
                        "INVOCATION_FILE": fpath_invocation,
                    }
                ],
            },
        ],
    )


def test_run_setup(config: Config, tmp_path: Path):
    runner = PipelineRunner(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="dummy_pipeline",
        pipeline_version="1.0.0",
    )
    create_empty_dataset(runner.dpath_root)
    config.save(runner.layout.fpath_config)
    runner.run_setup()
    assert runner.dpath_pipeline_output.exists()
    assert runner.dpath_pipeline_work.exists()


@pytest.mark.parametrize("keep_workdir", [True, False])
def test_run_cleanup(tmp_path: Path, keep_workdir):
    runner = PipelineRunner(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="dummy_pipeline",
        pipeline_version="1.0.0",
        keep_workdir=keep_workdir,
    )
    dpaths = [runner.dpath_pipeline_bids_db, runner.dpath_pipeline_work]
    for dpath in dpaths:
        dpath.mkdir(parents=True)
    runner.run_cleanup()
    for dpath in dpaths:
        if keep_workdir:
            assert dpath.exists()
        else:
            assert not dpath.exists()


@pytest.mark.parametrize("n_success", [1, 2])
def test_run_failed_cleanup(tmp_path: Path, n_success, config: Config):
    runner = PipelineRunner(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="dummy_pipeline",
        pipeline_version="1.0.0",
        keep_workdir=False,
    )
    runner.n_success = n_success
    runner.n_total = 2
    config.save(runner.layout.fpath_config)
    dpaths = [runner.dpath_pipeline_bids_db, runner.dpath_pipeline_work]
    for dpath in dpaths:
        dpath.mkdir(parents=True)
    runner.run_cleanup()
    if runner.n_success == runner.n_total:
        assert not dpath.exists()
    else:
        assert dpath.exists()


@pytest.mark.parametrize("simulate", [True, False])
def test_launch_boutiques_run(simulate, config: Config, tmp_path: Path):
    runner = PipelineRunner(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="dummy_pipeline",
        pipeline_version="1.0.0",
        simulate=simulate,
    )
    config.save(runner.layout.fpath_config)

    participant_id = "01"
    session_id = "BL"

    fids.create_fake_bids_dataset(
        runner.layout.dpath_bids,
        subjects=participant_id,
        sessions=session_id,
    )

    runner.dpath_pipeline_output.mkdir(parents=True, exist_ok=True)
    runner.dpath_pipeline_work.mkdir(parents=True, exist_ok=True)
    descriptor_str, invocation_str = runner.launch_boutiques_run(
        participant_id, session_id, container_command=""
    )

    assert "[[NIPOPPY_DPATH_BIDS]]" not in descriptor_str
    assert "[[NIPOPPY_PARTICIPANT_ID]]" not in invocation_str
    assert "[[NIPOPPY_BIDS_SESSION_ID]]" not in invocation_str


def test_process_container_config_boutiques_subcommand(config: Config, tmp_path: Path):
    # check that the container subcommand from the Boutiques container config is used
    runner = PipelineRunner(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="dummy_pipeline",
        pipeline_version="1.0.0",
    )

    config.save(runner.layout.fpath_config)

    participant_id = "01"
    session_id = "BL"

    # the container command in the config is "echo"
    # because otherwise the check for the container command fails
    # if Singularity/Apptainer is not on the PATH
    assert (
        runner.process_container_config(
            participant_id=participant_id, session_id=session_id
        )
        == "echo exec"
    )


@pytest.mark.parametrize(
    "doughnut_data,bagel_data,pipeline_name,pipeline_version,pipeline_step,expected",
    [
        (
            [
                ["01", "1", False],
                ["01", "2", True],
                ["01", "3", True],
            ],
            None,
            "dummy_pipeline",
            "1.0.0",
            "step1",
            [("01", "2"), ("01", "3")],
        ),
        (
            [
                ["01", "1", False],
                ["01", "2", True],
                ["01", "3", True],
            ],
            [],
            "dummy_pipeline",
            "1.0.0",
            "step1",
            [("01", "2"), ("01", "3")],
        ),
        (
            [
                ["01", "1", False],
                ["01", "2", True],
                ["01", "3", True],
            ],
            [
                ["01", "1", "dummy_pipeline", "1.0.0", "step1", Bagel.status_success],
                ["01", "2", "dummy_pipeline", "1.0.0", "step1", Bagel.status_success],
                ["01", "3", "dummy_pipeline", "1.0.0", "step1", Bagel.status_success],
            ],
            "dummy_pipeline",
            "1.0.0",
            "step1",
            [],
        ),
        (
            [
                ["01", "1", True],
                ["01", "2", True],
                ["01", "3", True],
            ],
            [
                ["01", "1", "dummy_pipeline", "1.0.0", "step1", Bagel.status_fail],
                ["01", "2", "dummy_pipeline", "1.0.0", "step1", Bagel.status_success],
                ["01", "3", "dummy_pipeline", "1.0.0", "step1", Bagel.status_fail],
                ["01", "1", "dummy_pipeline", "2.0", "step1", Bagel.status_success],
            ],
            "dummy_pipeline",
            "1.0.0",
            "step1",
            [("01", "1"), ("01", "3")],
        ),
        (
            [
                ["01", "1", True],
                ["01", "2", True],
                ["01", "3", True],
            ],
            [
                ["01", "1", "dummy_pipeline", "1.0.0", "step1", Bagel.status_fail],
                ["01", "2", "dummy_pipeline", "1.0.0", "step1", Bagel.status_success],
                ["01", "3", "dummy_pipeline", "1.0.0", "step1", Bagel.status_fail],
                ["01", "1", "dummy_pipeline", "1.0.0", "step2", Bagel.status_success],
                ["01", "2", "dummy_pipeline", "1.0.0", "step2", Bagel.status_success],
                ["01", "3", "dummy_pipeline", "1.0.0", "step2", Bagel.status_fail],
                ["01", "1", "dummy_pipeline", "2.0", "step1", Bagel.status_success],
            ],
            "dummy_pipeline",
            "1.0.0",
            "step2",
            [("01", "3")],
        ),
    ],
)
def test_get_participants_sessions_to_run(
    doughnut_data,
    bagel_data,
    pipeline_name,
    pipeline_version,
    pipeline_step,
    expected,
    config: Config,
    tmp_path: Path,
):
    participant_id = None
    session_id = None
    runner = PipelineRunner(
        dpath_root=tmp_path,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        pipeline_step=pipeline_step,
        participant_id=participant_id,
        session_id=session_id,
    )
    config.save(runner.layout.fpath_config)
    runner.doughnut = Doughnut().add_or_update_records(
        records=[
            {
                Doughnut.col_participant_id: data[0],
                Doughnut.col_session_id: data[1],
                Doughnut.col_visit_id: data[1],
                Doughnut.col_in_bids: data[2],
                Doughnut.col_datatype: None,
                Doughnut.col_participant_dicom_dir: "",
                Doughnut.col_in_pre_reorg: False,
                Doughnut.col_in_post_reorg: False,
            }
            for data in doughnut_data
        ]
    )
    if bagel_data is not None:
        Bagel(
            bagel_data,
            columns=[
                Bagel.col_participant_id,
                Bagel.col_session_id,
                Bagel.col_pipeline_name,
                Bagel.col_pipeline_version,
                Bagel.col_pipeline_step,
                Bagel.col_status,
            ],
        ).validate().save_with_backup(runner.layout.fpath_imaging_bagel)

    assert [
        tuple(x)
        for x in runner.get_participants_sessions_to_run(
            participant_id=participant_id, session_id=session_id
        )
    ] == expected


def test_run_multiple(config: Config, tmp_path: Path):
    participant_id = None
    session_id = None
    runner = PipelineRunner(
        dpath_root=tmp_path,
        pipeline_name="dummy_pipeline",
        pipeline_version="1.0.0",
        participant_id=participant_id,
        session_id=session_id,
    )
    config.save(runner.layout.fpath_config)

    participants_and_sessions = {"01": ["1"], "02": ["2"]}
    create_empty_dataset(runner.layout.dpath_root)
    manifest = prepare_dataset(
        participants_and_sessions_manifest=participants_and_sessions,
        participants_and_sessions_bidsified=participants_and_sessions,
        dpath_bidsified=runner.layout.dpath_bids,
    )
    manifest.save_with_backup(runner.layout.fpath_manifest)
    runner.run_setup()
    runner.run_main()

    bids_layout = BIDSLayout(database_path=runner.dpath_pipeline_bids_db)
    assert not len(bids_layout.get(extension=".nii.gz")) == 0


@pytest.mark.parametrize("generate_pybids_database", [True, False])
def test_run_single_pybidsdb(
    generate_pybids_database: bool,
    config: Config,
    mocker: pytest_mock.MockFixture,
    tmp_path: Path,
):
    participant_id = "01"
    session_id = "1"
    runner = PipelineRunner(
        dpath_root=tmp_path,
        pipeline_name="dummy_pipeline",
        pipeline_version="1.0.0",
    )
    config.save(runner.layout.fpath_config)

    # Set GENERATE_PYBIDS_DATABASE
    runner.pipeline_step_config.GENERATE_PYBIDS_DATABASE = generate_pybids_database

    # Mock the set_up_bids_db method
    mocked_set_up_bids_db = mocker.patch.object(runner, "set_up_bids_db")

    # Call run_single
    runner.run_single(participant_id=participant_id, session_id=session_id)

    # Assert set_up_bids_db was called or not called as expected
    if generate_pybids_database:
        mocked_set_up_bids_db.assert_called_once_with(
            dpath_pybids_db=runner.dpath_pipeline_bids_db,
            participant_id=participant_id,
            session_id=session_id,
        )
    else:
        mocked_set_up_bids_db.assert_not_called()
