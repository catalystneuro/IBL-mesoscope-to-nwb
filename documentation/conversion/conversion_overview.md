# Conversion Overview

This document describes how to run a conversion, the shape of the Python API,
and the support utilities for checking data availability, pre-downloading
files, running in stub-test mode, and converting many sessions in parallel.

## Authentication

Both pipelines use the ONE API for remote data access. You must have valid IBL
credentials configured locally. By default, `ONE()` reads credentials from your
local ONE configuration. To connect to a specific Alyx server:

```python
from one.api import ONE
one = ONE(base_url="https://alyx.internationalbrainlab.org")
```

## Quick Start

Convert a single session in both raw and processed modes:

```python
from pathlib import Path
from one.api import ONE
from ibl_mesoscope_to_nwb.mesoscope2025.conversion import (
    convert_raw_session,
    convert_processed_session,
)

eid = "5ce2e17e-8471-42d4-8a16-21949710b328"
one = ONE()
output_path = Path("/data/IBL-mesoscope-nwbfiles")

# Raw acquisition data (ScanImage TIFFs, DAQ, videos, visual stimulus)
result_raw = convert_raw_session(
    eid=eid,
    one=one,
    output_path=output_path,
    verbose=True,
)
print(f"Raw NWB: {result_raw['nwbfile_path']} ({result_raw['nwb_size_gb']:.2f} GB)")

# Processed / analyzed data (motion correction, segmentation, behavior)
result_processed = convert_processed_session(
    eid=eid,
    one=one,
    output_path=output_path,
    verbose=True,
)
print(f"Processed NWB: {result_processed['nwbfile_path']}")
```

Both functions return a dict:

| Key | Type | Description |
|-----|------|-------------|
| `nwbfile_path` | `Path` | Absolute path to the written NWB file |
| `nwb_size_bytes` | `int` | NWB file size in bytes |
| `nwb_size_gb` | `float` | NWB file size in gigabytes |
| `write_time` | `float` | Time taken to write the file (seconds) |

## Entry points

### `convert_raw_session`

```python
from ibl_mesoscope_to_nwb.mesoscope2025.conversion import convert_raw_session

result = convert_raw_session(
    eid="5ce2e17e-8471-42d4-8a16-21949710b328",
    one=ONE(),
    output_path=Path("/data/IBL-mesoscope-nwbfiles"),
    stub_test=False,                # True for a quick test run
    append_on_disk_nwbfile=False,   # True to append to an existing NWB file
    verbose=True,
)
```

Defined in
[src/ibl_mesoscope_to_nwb/mesoscope2025/conversion/raw.py](../../src/ibl_mesoscope_to_nwb/mesoscope2025/conversion/raw.py).

### `convert_processed_session`

```python
from ibl_mesoscope_to_nwb.mesoscope2025.conversion import convert_processed_session

result = convert_processed_session(
    eid="5ce2e17e-8471-42d4-8a16-21949710b328",
    one=ONE(),
    output_path=Path("/data/IBL-mesoscope-nwbfiles"),
    stub_test=False,
    append_on_disk_nwbfile=False,
    verbose=True,
)
```

Defined in
[src/ibl_mesoscope_to_nwb/mesoscope2025/conversion/processed.py](../../src/ibl_mesoscope_to_nwb/mesoscope2025/conversion/processed.py).

### `session_to_nwb`

A unified wrapper around both pipelines:

```python
from ibl_mesoscope_to_nwb.mesoscope2025.convert_session import session_to_nwb

session_to_nwb(
    output_path=Path("/data/IBL-mesoscope-nwbfiles"),
    eid="5ce2e17e-8471-42d4-8a16-21949710b328",
    mode="raw",        # or "processed"
    verbose=True,
)
```

## Querying a session before converting

The pipeline automatically determines the number of FOVs from session metadata,
but you can query this independently:

```python
from ibl_mesoscope_to_nwb.mesoscope2025.utils import (
    get_number_of_FOVs_from_raw_imaging_metadata,
    get_FOV_names_from_alf_collections,
    get_available_tasks_from_alf_collections,
)
from one.api import ONE

one = ONE()
eid = "5ce2e17e-8471-42d4-8a16-21949710b328"

n_fovs = get_number_of_FOVs_from_raw_imaging_metadata(one, eid)
print(f"Session has {n_fovs} FOVs")           # e.g. 8

fov_names = get_FOV_names_from_alf_collections(one, eid)
print(fov_names)                               # ["FOV_00", "FOV_01", …]

task_names = get_available_tasks_from_alf_collections(one, eid)
print(task_names)                              # ["task_00", "task_01"]
```

## Checking Data Availability

Every interface exposes a `check_availability` classmethod that queries the ONE
API without downloading any files. Use this before a conversion to inspect
what data is present.

```python
from one.api import ONE
from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import (
    MesoscopeSegmentationInterface,
    MesoscopeROIAnatomicalLocalizationInterface,
    MesoscopeDAQInterface,
    TaskSettingsInterface,
    VisualStimulusInterface,
)

one = ONE()
eid = "5ce2e17e-8471-42d4-8a16-21949710b328"

# Interfaces that require a FOV name
result = MesoscopeSegmentationInterface.check_availability(one, eid, FOV_name="FOV_00")
print(result["available"])          # True / False
print(result["missing_required"])   # list of missing file patterns, if any

result = MesoscopeROIAnatomicalLocalizationInterface.check_availability(
    one, eid, FOV_name="FOV_00"
)

# Interfaces that do not require a FOV name
MesoscopeDAQInterface.check_availability(one, eid)
TaskSettingsInterface.check_availability(one, eid)
VisualStimulusInterface.check_availability(one, eid)
```

The returned dict always includes:

- `available` (`bool`) — whether all required data is present.
- `missing_required` (`list`) — file patterns required but not found.
- `found_files` (`list`) — file patterns that were successfully located.

## Downloading Data

Every interface also exposes a `download_data` classmethod. Call this to
pre-download files to the local ONE cache before starting a conversion — useful
when running conversions on a machine without a direct internet connection
during the conversion step itself.

```python
from one.api import ONE
from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import (
    MesoscopeDAQInterface,
    MesoscopeSegmentationInterface,
    MesoscopeROIAnatomicalLocalizationInterface,
    MesoscopeImageAnatomicalLocalizationInterface,
    TaskSettingsInterface,
    VisualStimulusInterface,
)

one = ONE()
eid = "5ce2e17e-8471-42d4-8a16-21949710b328"

MesoscopeDAQInterface.download_data(one, eid)
TaskSettingsInterface.download_data(one, eid)
VisualStimulusInterface.download_data(one, eid)

for fov_name in ["FOV_00", "FOV_01"]:
    MesoscopeSegmentationInterface.download_data(one, eid, FOV_name=fov_name)
    MesoscopeROIAnatomicalLocalizationInterface.download_data(one, eid, FOV_name=fov_name)
    MesoscopeImageAnatomicalLocalizationInterface.download_data(one, eid, FOV_name=fov_name)
```

## Stub Test Mode

Stub-test mode converts a small subset of the data, making it suitable for
rapid testing without requiring full data downloads or long processing times.

```python
from pathlib import Path
from one.api import ONE
from ibl_mesoscope_to_nwb.mesoscope2025.conversion import (
    convert_raw_session,
    convert_processed_session,
)

one = ONE()
eid = "5ce2e17e-8471-42d4-8a16-21949710b328"
output_path = Path("/tmp/nwb-stub-test")

convert_raw_session(eid=eid, one=one, output_path=output_path, stub_test=True)
convert_processed_session(eid=eid, one=one, output_path=output_path, stub_test=True)
```

In stub mode the following limits apply:

| Data | Limit |
|------|-------|
| FOVs | 2 (instead of all available, typically 8) |
| DAQ samples | 10,000 |
| Trial table | 10 trials |
| Raw camera videos | Skipped unless already in the local ONE cache |

Stub output files are written under `{output_path}/stub/` rather than
`{output_path}/full/`.

## Parallel conversion of multiple sessions

`session_to_nwb` unifies both pipelines behind a single function; `dataset_to_nwb`
wraps it with a `ProcessPoolExecutor` for parallel execution.

### Sequential

```python
from pathlib import Path
from ibl_mesoscope_to_nwb.mesoscope2025.convert_session import session_to_nwb

eids = [
    "5ce2e17e-8471-42d4-8a16-21949710b328",
    "42d7e11e-3185-4a79-a6ad-bbaf47366db2",
    "4693e7cc-17f6-4eeb-8abb-5951ba82b601",
]
output_path = Path("/data/IBL-mesoscope-nwbfiles")

for eid in eids:
    for mode in ("raw", "processed"):
        try:
            session_to_nwb(output_path=output_path, eid=eid, mode=mode, verbose=True)
        except Exception as e:
            print(f"Failed for {eid} ({mode}): {e}")
```

### Parallel

```python
from pathlib import Path
from ibl_mesoscope_to_nwb.mesoscope2025.convert_all_sessions import dataset_to_nwb

dataset_to_nwb(
    data_dir_path=Path("/data/IBL-raw"),
    output_dir_path=Path("/data/IBL-mesoscope-nwbfiles"),
    max_workers=4,
    verbose=True,
)
```

Each worker calls `safe_session_to_nwb`, which catches exceptions and writes
them to an `.error` file on disk rather than raising, so a failure in one
session does not abort the entire dataset conversion.

**Note:** `get_session_to_nwb_kwargs_per_session` in `convert_all_sessions.py`
is a placeholder that raises `NotImplementedError`. You must implement this
function to return a list of per-session kwargs dicts appropriate for your
dataset.

## Reading a converted NWB file

```python
from pynwb import NWBHDF5IO

nwbfile_path = (
    "/data/IBL-mesoscope-nwbfiles/full/sub-SWC054/"
    "sub-SWC054_ses-5ce2e17e-8471-42d4-8a16-21949710b328_desc-processed_behavior+ophys.nwb"
)

with NWBHDF5IO(nwbfile_path, "r") as io:
    nwbfile = io.read()

    # ROI fluorescence traces for the first FOV
    ophys = nwbfile.processing["ophys"]
    fluorescence = ophys["Fluorescence"]["RoiResponseSeriesFOV00"]
    fluorescence.data[:]          # shape: (n_frames, n_rois)
    fluorescence.timestamps[:]    # seconds

    # Trial table
    trials = nwbfile.trials.to_dataframe()

    # Wheel position
    behavior = nwbfile.processing["behavior"]
    wheel_pos = behavior["WheelPosition"]

    # Anatomical localization
    localization = nwbfile.lab_meta_data["localization"]
```

See the [tutorial notebooks](../tutorials.md) for a full walkthrough of each
pipeline's output.
