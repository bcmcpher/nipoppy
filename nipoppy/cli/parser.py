"""Parsers for the CLI."""

import logging
from argparse import ArgumentParser, HelpFormatter, _ActionsContainer, _SubParsersAction
from pathlib import Path

from nipoppy._version import __version__
from nipoppy.env import BIDS_SESSION_PREFIX, BIDS_SUBJECT_PREFIX

PROGRAM_NAME = "nipoppy"
COMMAND_INIT = "init"
COMMAND_STATUS = "status"
COMMAND_DOUGHNUT = "doughnut"
COMMAND_DICOM_REORG = "reorg"
COMMAND_BIDS_CONVERSION = "bidsify"
COMMAND_PIPELINE_RUN = "run"
COMMAND_PIPELINE_TRACK = "track"
COMMAND_PIPELINE_EXTRACT = "extract"

DEFAULT_VERBOSITY = "2"  # info
VERBOSITY_TO_LOG_LEVEL_MAP = {
    "0": logging.ERROR,
    "1": logging.WARNING,
    "2": logging.INFO,
    "3": logging.DEBUG,
}


def add_arg_dataset_root(parser: _ActionsContainer) -> _ActionsContainer:
    """Add a dataset-root argument to the parser."""
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the root of the dataset.",
    )
    return parser


def add_args_participant_and_session(parser: _ActionsContainer) -> _ActionsContainer:
    """Add --participant-id and --session-id arguments to the parser."""
    parser.add_argument(
        "--participant-id",
        required=False,
        help=f"Participant ID (with or without the {BIDS_SUBJECT_PREFIX} prefix).",
    )
    parser.add_argument(
        "--session-id",
        required=False,
        help=f"Session ID (with or without the {BIDS_SESSION_PREFIX} prefix).",
    )
    return parser


def add_args_pipeline(parser: _ActionsContainer) -> _ActionsContainer:
    """Add pipeline-related arguments to the parser."""
    parser.add_argument(
        "--pipeline",
        dest="pipeline_name",
        type=str,
        required=True,
        help="Pipeline name, as specified in the config file.",
    )
    parser.add_argument(
        "--pipeline-version",
        type=str,
        required=False,
        help=(
            "Pipeline version, as specified in the config file "
            "(default: first version)."
        ),
    )
    parser.add_argument(
        "--pipeline-step",
        type=str,
        required=False,
        help="Pipeline step, as specified in the config file (default: first step).",
    )

    return parser


def add_arg_layout(parser: _ActionsContainer) -> _ActionsContainer:
    """Add a --layout argument to the parser."""
    parser.add_argument(
        "--layout",
        dest="fpath_layout",
        type=Path,
        required=False,
        help=(
            "Path to a custom layout specification file"
            ", to be used instead of the default layout."
        ),  # TODO point to example
    )


def add_arg_dry_run(parser: _ActionsContainer) -> _ActionsContainer:
    """Add a --dry-run argument to the parser."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands but do not execute them.",
    )
    return parser


def add_arg_help(parser: _ActionsContainer) -> _ActionsContainer:
    """Add a --help argument to the parser."""
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    return parser


def add_arg_version(parser: _ActionsContainer) -> _ActionsContainer:
    """Add a --version argument to the parser."""
    parser.add_argument(
        "--version",
        action="version",
        help="Show version number and exit.",
        version=f"{__version__}",
    )
    return parser


def add_arg_verbosity(parser: _ActionsContainer) -> _ActionsContainer:
    """Add a --verbosity argument to the parser."""
    parser.add_argument(
        "--verbosity",
        default=DEFAULT_VERBOSITY,
        choices=VERBOSITY_TO_LOG_LEVEL_MAP.keys(),
        help=(
            "Verbosity level, from 0 (least verbose) to 3 (most verbose)."
            f" Default: {DEFAULT_VERBOSITY}."
        ),
    )
    return parser


def add_args_runner(parser: _ActionsContainer) -> _ActionsContainer:
    """Add arguments for a runner."""
    parser = add_args_pipeline(parser)
    parser = add_args_participant_and_session(parser)
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate the pipeline run without executing the generated command-line.",
    )
    return parser


def add_subparser_init(
    subparsers: _SubParsersAction,
    formatter_class: type[HelpFormatter] = HelpFormatter,
) -> ArgumentParser:
    """Add subparser for init command."""
    description = "Initialize a new dataset."
    parser = subparsers.add_parser(
        COMMAND_INIT,
        description=description,
        help=description,
        formatter_class=formatter_class,
        add_help=False,
    )
    parser = add_arg_dataset_root(parser)
    parser.add_argument(
        "--bids-source",
        type=str,
        required=False,
        help=(
            "Path to an existing BIDS dataset to initialize the new Nipoppy dataset "
            "with. Note: this will automatically create an appropriate manifest file."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        choices=("copy", "move"),
        help=(
            "If using a BIDS source, specify whether to copy or move the files "
            "(default: copy)."
        ),
    )

    return parser


def add_subparser_status(
    subparsers: _SubParsersAction,
    formatter_class: type[HelpFormatter] = HelpFormatter,
) -> ArgumentParser:
    """Add subparser for status command."""
    description = "Show current status of the dataset."
    parser = subparsers.add_parser(
        COMMAND_STATUS,
        description=description,
        help=description,
        formatter_class=formatter_class,
        add_help=False,
    )

    parser = add_arg_dataset_root(parser)

    return parser


def add_subparser_doughnut(
    subparsers: _SubParsersAction,
    formatter_class: type[HelpFormatter] = HelpFormatter,
) -> ArgumentParser:
    """Add subparser for doughnut command."""
    description = "Create/update a dataset's doughnut file."
    parser = subparsers.add_parser(
        COMMAND_DOUGHNUT,
        description=description,
        help=description,
        formatter_class=formatter_class,
        add_help=False,
    )
    parser = add_arg_dataset_root(parser)
    parser.add_argument(
        "--empty",
        action="store_true",
        help=(
            "Set all statuses to False in newly added records"
            " (regardless of what is on disk). May be useful to reduce runtime."
        ),
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help=(
            "Regenerate the doughnut file even if it already exists"
            " (default: only append rows for new records)"
        ),
    )
    return parser


def add_subparser_reorg(
    subparsers: _SubParsersAction,
    formatter_class: type[HelpFormatter] = HelpFormatter,
) -> ArgumentParser:
    """Add subparser for reorg command."""
    from nipoppy.layout import DEFAULT_LAYOUT_INFO

    description = (
        "(Re)organize raw (DICOM) files, from the "
        f"({DEFAULT_LAYOUT_INFO.dpath_pre_reorg}) to "
        f"({DEFAULT_LAYOUT_INFO.dpath_post_reorg})."
    )
    parser = subparsers.add_parser(
        COMMAND_DICOM_REORG,
        description=description,
        help=description,
        formatter_class=formatter_class,
        add_help=False,
    )
    parser = add_arg_dataset_root(parser)
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy files when reorganizing (default: create symlinks).",
    )
    parser.add_argument(
        "--check-dicoms",
        action="store_true",
        help=(
            "Read DICOM file headers when reorganizing and check if they have the "
            '"DERIVED" image type (which can be problematic for some BIDS '
            "converters). The paths to the derived DICOMs will be written to the log."
        ),
    )
    return parser


def add_subparser_bidsify(
    subparsers: _SubParsersAction, formatter_class: type[HelpFormatter] = HelpFormatter
) -> ArgumentParser:
    """Add subparser for run command."""
    description = "Run a BIDS conversion pipeline."
    parser = subparsers.add_parser(
        COMMAND_BIDS_CONVERSION,
        description=description,
        help=description,
        formatter_class=formatter_class,
        add_help=False,
    )
    parser = add_arg_dataset_root(parser)
    parser = add_args_runner(parser)
    return parser


def add_subparser_run(
    subparsers: _SubParsersAction, formatter_class: type[HelpFormatter] = HelpFormatter
) -> ArgumentParser:
    """Add subparser for run command."""
    description = "Run a processing pipeline."
    parser = subparsers.add_parser(
        COMMAND_PIPELINE_RUN,
        description=description,
        help=description,
        formatter_class=formatter_class,
        add_help=False,
    )
    parser = add_arg_dataset_root(parser)
    parser = add_args_runner(parser)
    parser.add_argument(
        "--keep-workdir",
        type=str,
        required=False,
        help=(
            "Keep pipeline working directory upon success "
            "(default: working directory deleted unless a run failed)."
        ),
    )
    parser.add_argument(
        "--tar",
        action="store_true",
        help=(
            "Archive participant-session-level results into a tarball upon "
            "successful completion. The path to be archived should be specified "
            "in the tracker configuration file."
        ),
    )
    return parser


def add_subparser_track(
    subparsers: _SubParsersAction, formatter_class: type[HelpFormatter] = HelpFormatter
) -> ArgumentParser:
    """Add subparser for track command."""
    description = "Track the processing status of a pipeline."
    parser = subparsers.add_parser(
        COMMAND_PIPELINE_TRACK,
        description=description,
        help=description,
        formatter_class=formatter_class,
        add_help=False,
    )
    parser = add_arg_dataset_root(parser)
    parser = add_args_pipeline(parser)
    parser = add_args_participant_and_session(parser)
    return parser


def add_subparser_extract(
    subparsers: _SubParsersAction, formatter_class: type[HelpFormatter] = HelpFormatter
) -> ArgumentParser:
    """Add subparser for extract command."""
    description = "Run an extraction pipeline."
    parser = subparsers.add_parser(
        COMMAND_PIPELINE_EXTRACT,
        description=description,
        help=description,
        formatter_class=formatter_class,
        add_help=False,
    )
    parser = add_arg_dataset_root(parser)
    parser = add_args_runner(parser)

    return parser


def get_global_parser(
    formatter_class: type[HelpFormatter] = HelpFormatter,
) -> ArgumentParser:
    """Get the global parser."""
    global_parser = ArgumentParser(
        prog=PROGRAM_NAME,
        description="Organize and process neuroimaging-clinical datasets.",
        epilog=(
            f"Run '{PROGRAM_NAME} COMMAND --help'"
            " for more information on a subcommand."
        ),
        formatter_class=formatter_class,
        add_help=False,
    )
    add_arg_help(global_parser)
    add_arg_version(global_parser)

    # subcommand parsers
    subparsers = global_parser.add_subparsers(
        title="Subcommands",
        dest="command",
        required=True,
    )
    add_subparser_init(subparsers, formatter_class=formatter_class)
    add_subparser_status(subparsers, formatter_class=formatter_class)
    add_subparser_doughnut(subparsers, formatter_class=formatter_class)
    add_subparser_reorg(subparsers, formatter_class=formatter_class)
    add_subparser_bidsify(subparsers, formatter_class=formatter_class)
    add_subparser_run(subparsers, formatter_class=formatter_class)
    add_subparser_track(subparsers, formatter_class=formatter_class)
    add_subparser_extract(subparsers, formatter_class=formatter_class)

    # add common/global options to subcommand parsers
    for parser in list(subparsers.choices.values()):
        common_arg_group = parser.add_argument_group("Global options")
        add_arg_layout(common_arg_group)
        add_arg_verbosity(common_arg_group)
        add_arg_dry_run(common_arg_group)
        add_arg_help(common_arg_group)

    return global_parser
