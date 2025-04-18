# Glossary

## Nipoppy terms

```{glossary}
Doughnut file
    A tabular file at {{fpath_doughnut}} that keeps track of the status of raw imaging data (i.e., whether it is available and/or has been reorganized and/or has been converted to {term}`BIDS`). This file is automatically generated/updated by Nipoppy workflows, though it can safely be deleted manually (if e.g. it contains outdated information), in which case it will be regenerated automatically when needed. The doughnut file can also be created/updated with the [`nipoppy doughnut`](./cli_reference/doughnut.rst) command if needed.

    See {ref}`here <doughnut-schema>` for more information about the columns in the file.

Imaging bagel file
    A tabular file at {{fpath_imaging_bagel}} that indicates the completion status of processing pipelines of interest at the participant-session level. The bagel file is created by the [`nipoppy track`](./cli_reference/track.rst) command and can be used as input to [the Neurobagel CLI](https://neurobagel.org/user_guide/cli).

    See {ref}`here <imaging-bagel-schema>` for more information about the columns in the file.

Session ID
    A BIDS-compliant session identifier, without the `"ses-"` prefix.

Visit ID
    An identifier for a data collection event, not restricted to imaging data.
```

### Session IDs vs visit IDs

Nipoppy uses the term "session ID" for imaging data, following the convention established by BIDS. The term "visit ID", on the other hand, is used to refer to any data collection event (not necessarily imaging-related), and is more common in clinical contexts. In most cases, `session_id` and `visit_id` will be identical (or `session_id`s will be a subset of `visit_id`s). However, having two descriptors becomes particularly useful when imaging and non-imaging assessments do not use the same naming conventions.

## Neuroimaging/software terms

```{glossary}

API
    Application Programming Interface, how software interacts with other software.

BIDS
    The Brain Imaging Data Structure, a community standard for organizing neuroimaging (and other) data. See the [BIDS website](https://bids.neuroimaging.io/) for more information.

Boutiques
    A flexible framework for describing and executing command-line tools. Boutiques is based on JSON *descriptor* files that list tool inputs, outputs, error codes, and more. JSON *invocation* files are used to specify runtime parameters. See the [website](https://boutiques.github.io/) for more information.

CLI
    Command-line interface, i.e. software that can be run in the Terminal.

`conda`
    An package and environment manager for Python (and other) environments. See the [`conda` website](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) for more information.

HPC
    High-perfomance computing system, i.e. a compute cluster or supercomputer.

JSON
    JavaScript Object Notation, a file format for storing and sharing data. JSON structures are combinations of *objects* (key-value pairs) and *arrays* (ordered lists). See the [website](https://www.json.org/json-en.html) for more information.

MRI
    Magnetic resonance imaging, the most widely used neuroimaging modality.

PyPI
    The [Python Package Index](https://pypi.org/), a repository of Python packages that are `pip`-installable.

`venv`
    A Python built-in library for creating Python virtual environments. See the [Python documentation](https://docs.python.org/3/library/venv.html) for more information.
```
