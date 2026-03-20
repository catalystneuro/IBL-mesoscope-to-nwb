# IBL Mesoscope NWB Conversion Pipeline — Usage Guide

This document describes how to use the IBL mesoscope NWB conversion pipeline to convert IBL mesoscope experimental data into NWB format.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Pipeline Overview](#pipeline-overview)
4. [Raw Pipeline](#raw-pipeline)
5. [Processed Pipeline](#processed-pipeline)
6. [Checking Data Availability](#checking-data-availability)
7. [Downloading Data](#downloading-data)
8. [Stub Test Mode](#stub-test-mode)
9. [Parallel Conversion of Multiple Sessions](#parallel-conversion-of-multiple-sessions)
10. [NWB File Structure](#nwb-file-structure)
11. [Anatomical Localization Coordinate Spaces](#anatomical-localization-coordinate-spaces)

---

## Installation

Clone the repository and install the `mesoscope2025` extras:

```bash
git clone https://github.com/catalystneuro/IBL-mesoscope-to-nwb.git
cd IBL-mesoscope-to-nwb
pip install -e ".[mesoscope2025]"
```

The `mesoscope2025` extra pins the following versions for reproducibility:

- `neuroconv==0.8.1`
- `roiextractors==0.7.0`
- `sparse` (required for loading IBL sparse mask format)

---

## Quick Start

Convert a single session in both raw and processed modes:

```python
from pathlib import Path
from one.api import ONE
from ibl_mesoscope_to_nwb.mesoscope2025.conversion import convert_raw_session, convert_processed_session

eid = "5ce2e17e-8471-42d4-8a16-21949710b328"
one = ONE()  # authenticates with your local ONE credentials
output_path = Path("/data/IBL-mesoscope-nwbfiles")

# Convert raw acquisition data
result_raw = convert_raw_session(
    eid=eid,
    one=one,
    output_path=output_path,
    verbose=True,
)
print(f"Raw NWB written to: {result_raw['nwbfile_path']}")
print(f"File size: {result_raw['nwb_size_gb']:.2f} GB, write time: {result_raw['write_time']:.1f}s")

# Convert processed/analyzed data
result_processed = convert_processed_session(
    eid=eid,
    one=one,
    output_path=output_path,
    verbose=True,
)
print(f"Processed NWB written to: {result_processed['nwbfile_path']}")
```

Both functions return a dict with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `nwbfile_path` | `Path` | Absolute path to the written NWB file |
| `nwb_size_bytes` | `int` | NWB file size in bytes |
| `nwb_size_gb` | `float` | NWB file size in gigabytes |
| `write_time` | `float` | Time taken to write the file (seconds) |

---

## Pipeline Overview

There are two independent conversion pipelines, each producing a separate BIDS-compliant NWB file:

| Pipeline | Function | Output filename |
|----------|----------|-----------------|
| Raw | `convert_raw_session` | `sub-{subject}_ses-{eid}_desc-raw_behavior+ophys.nwb` |
| Processed | `convert_processed_session` | `sub-{subject}_ses-{eid}_desc-processed_behavior+ophys.nwb` |

Output files are written to:

```
{output_path}/{full|stub}/sub-{subject}/sub-{subject}_ses-{eid}_desc-{raw|processed}_behavior+ophys.nwb
```

`full` is used for standard conversions; `stub` is used when `stub_test=True`.

### Authentication

Both pipelines use the ONE API for remote data access. You must have valid IBL credentials configured. By default, `ONE()` reads credentials from your local ONE configuration. To connect to a specific Alyx server:

```python
one = ONE(base_url="https://alyx.internationalbrainlab.org")
```

---

## Raw Pipeline

The raw pipeline converts data as acquired, without any post-processing. It uses the following data interfaces:

| Interface | Data |
|-----------|------|
| `MesoscopeRawImagingInterface` | ScanImage TIFF files, one per field of view (FOV) |
| `MesoscopeDAQInterface` | Timeline DAQ board analog and digital channels |
| `TaskSettingsInterface` | Session epoch timing from `_iblrig_taskSettings.raw.json` |
| `VisualStimulusInterface` | Passive visual stimulus video (`_sp_video.mp4`) and presentation intervals |
| `RawVideoInterface` | Raw behavioral camera videos (left, right, body cameras) |

### Running the raw pipeline

```python
from pathlib import Path
from one.api import ONE
from ibl_mesoscope_to_nwb.mesoscope2025.conversion import convert_raw_session

result = convert_raw_session(
    eid="5ce2e17e-8471-42d4-8a16-21949710b328",
    one=ONE(),
    output_path=Path("/data/IBL-mesoscope-nwbfiles"),
    stub_test=False,         # set True for a quick test run
    append_on_disk_nwbfile=False,  # set True to append to an existing NWB file
    verbose=True,
)
```

### Number of FOVs

The pipeline automatically determines the number of FOVs from session metadata. You can query this independently:

```python
from ibl_mesoscope_to_nwb.mesoscope2025.utils import get_number_of_FOVs_from_raw_imaging_metadata
from one.api import ONE

n_fovs = get_number_of_FOVs_from_raw_imaging_metadata(ONE(), eid)
print(f"Session has {n_fovs} FOVs")
```

---

## Processed Pipeline

The processed pipeline converts analyzed outputs. It uses the following data interfaces:

| Interface | Data |
|-----------|------|
| `MesoscopeMotionCorrectedImagingInterface` | Motion-corrected binary imaging data, one per FOV |
| `MesoscopeSegmentationInterface` | ROI segmentation: masks, fluorescence traces, deconvolved traces |
| `MesoscopeROIAnatomicalLocalizationInterface` | Per-ROI coordinates in IBL-Bregma and Allen CCF v3 spaces |
| `MesoscopeImageAnatomicalLocalizationInterface` | Per-pixel mean image coordinates in IBL-Bregma and Allen CCF v3 spaces |
| `TaskSettingsInterface` | Session epoch timing |
| `MesoscopeWheelPositionInterface` | Wheel position time series per task |
| `MesoscopeWheelKinematicsInterface` | Wheel velocity/speed per task |
| `MesoscopeWheelMovementsInterface` | Detected wheel movements per task |
| `BrainwideMapTrialsInterface` | Trial table |
| `IblPoseEstimationInterface` | Pose estimation (Lightning Pose or DLC) per camera |
| `PupilTrackingInterface` | Pupil diameter tracking per camera |
| `RoiMotionEnergyInterface` | ROI motion energy per camera |

### Running the processed pipeline

```python
from pathlib import Path
from one.api import ONE
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

### Querying available FOVs and tasks

```python
from ibl_mesoscope_to_nwb.mesoscope2025.utils import (
    get_FOV_names_from_alf_collections,
    get_available_tasks_from_alf_collections,
)
from one.api import ONE

one = ONE()
eid = "5ce2e17e-8471-42d4-8a16-21949710b328"

fov_names = get_FOV_names_from_alf_collections(one, eid)
print(fov_names)  # e.g. ["FOV_00", "FOV_01", "FOV_02", ...]

task_names = get_available_tasks_from_alf_collections(one, eid)
print(task_names)  # e.g. ["task_00", "task_01"]
```

---

## Checking Data Availability

Every interface exposes a `check_availability` classmethod that queries the ONE API without downloading any files. This is useful for inspecting what data is present before starting a conversion.

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

# Check interfaces that require a FOV name
result = MesoscopeSegmentationInterface.check_availability(one, eid, FOV_name="FOV_00")
print(result)
# {
#   "available": True,
#   "missing_required": [],
#   "found_files": [...],
#   ...
# }

# Check anatomical localization for a specific FOV
result = MesoscopeROIAnatomicalLocalizationInterface.check_availability(one, eid, FOV_name="FOV_00")
print(result["available"])  # True or False
print(result["missing_required"])  # list of missing file patterns, if any

# Check interfaces that do not require a FOV name
result = MesoscopeDAQInterface.check_availability(one, eid)
result = TaskSettingsInterface.check_availability(one, eid)
result = VisualStimulusInterface.check_availability(one, eid)
```

The returned dict always includes:

- `available` (`bool`): whether all required data is present
- `missing_required` (`list`): file patterns that are required but not found
- `found_files` (`list`): file patterns that were successfully located

---

## Downloading Data

Every interface also exposes a `download_data` classmethod. Call this to pre-download files to the local ONE cache before starting a conversion, which is useful when running conversions on a machine without a direct internet connection during the conversion step.

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

# Download DAQ and task settings data
MesoscopeDAQInterface.download_data(one, eid)
TaskSettingsInterface.download_data(one, eid)
VisualStimulusInterface.download_data(one, eid)

# Download per-FOV data
for fov_name in ["FOV_00", "FOV_01"]:
    MesoscopeSegmentationInterface.download_data(one, eid, FOV_name=fov_name)
    MesoscopeROIAnatomicalLocalizationInterface.download_data(one, eid, FOV_name=fov_name)
    MesoscopeImageAnatomicalLocalizationInterface.download_data(one, eid, FOV_name=fov_name)
```

---

## Stub Test Mode

Stub test mode converts a small subset of the data, making it suitable for rapid testing of the pipeline without requiring full data downloads or long processing times.

```python
from pathlib import Path
from one.api import ONE
from ibl_mesoscope_to_nwb.mesoscope2025.conversion import convert_raw_session, convert_processed_session

one = ONE()
eid = "5ce2e17e-8471-42d4-8a16-21949710b328"
output_path = Path("/tmp/nwb-stub-test")

result = convert_raw_session(eid=eid, one=one, output_path=output_path, stub_test=True)
result = convert_processed_session(eid=eid, one=one, output_path=output_path, stub_test=True)
```

In stub mode the following limits apply:

| Data | Limit |
|------|-------|
| FOVs | 2 (instead of all available, typically 8) |
| DAQ samples | 10,000 |
| Trial table | 10 trials |
| Raw camera videos | Skipped unless the video file is already in the local ONE cache |

Stub output files are written under `{output_path}/stub/` rather than `{output_path}/full/`.

---

## Parallel Conversion of Multiple Sessions

The `session_to_nwb` helper function unifies both pipelines under a single interface, and a `dataset_to_nwb` wrapper handles parallel execution across sessions using `ProcessPoolExecutor`.

### Converting multiple sessions sequentially

```python
import time
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

### Converting a dataset in parallel

```python
from pathlib import Path
from ibl_mesoscope_to_nwb.mesoscope2025.convert_all_sessions import dataset_to_nwb

dataset_to_nwb(
    data_dir_path=Path("/data/IBL-raw"),
    output_dir_path=Path("/data/IBL-mesoscope-nwbfiles"),
    max_workers=4,   # number of parallel worker processes
    verbose=True,
)
```

`dataset_to_nwb` uses `ProcessPoolExecutor` for parallelism. Each worker calls `safe_session_to_nwb`, which catches exceptions and writes them to an `.error` file on disk rather than raising, so a failure in one session does not abort the entire dataset conversion.

**Note:** `get_session_to_nwb_kwargs_per_session` in `convert_all_sessions.py` is a placeholder that raises `NotImplementedError`. You must implement this function to return a list of per-session kwargs dicts appropriate for your dataset.

---

## NWB File Structure

The table below shows where each type of data lands inside the NWB file.

| Data | NWB type | NWB location |
|------|----------|--------------|
| Raw imaging (per FOV) | `TwoPhotonSeries` | `nwbfile.acquisition` |
| DAQ analog channels | `TimeSeries` | `nwbfile.acquisition` |
| DAQ digital channels | `LabeledEvents` (ndx-events) | `nwbfile.acquisition` |
| Raw behavioral videos | `ImageSeries` | `nwbfile.acquisition` |
| Visual stimulus video | `OpticalSeries` | `nwbfile.stimulus_templates` |
| Visual stimulus intervals | `TimeIntervals` | `nwbfile.stimulus` |
| Session/task epochs | `TimeIntervals` named `"epochs"` | `nwbfile.epochs` |
| Motion-corrected imaging (per FOV) | `TwoPhotonSeries` | `nwbfile.processing["ophys"]` |
| ROI segmentation (per FOV) | `PlaneSegmentation` + `RoiResponseSeries` | `nwbfile.processing["ophys"]` |
| Anatomical coords (ROI-level) | `AnatomicalCoordinatesTable` | `nwbfile.lab_meta_data["localization"]` |
| Anatomical coords (image-level) | `AnatomicalCoordinatesImage` | `nwbfile.lab_meta_data["localization"]` |
| Trials | `TimeIntervals` | `nwbfile.trials` |
| Wheel position | `SpatialSeries` | `nwbfile.processing["behavior"]` |
| Wheel kinematics | `TimeSeries` | `nwbfile.processing["behavior"]` |
| Wheel movements | `TimeIntervals` | `nwbfile.processing["behavior"]` |
| Pose estimation | `PoseEstimation` | `nwbfile.processing["behavior"]` |
| Pupil tracking | `TimeSeries` | `nwbfile.processing["behavior"]` |
| ROI motion energy | `TimeSeries` | `nwbfile.processing["behavior"]` |

### Accessing data from a written NWB file

```python
from pynwb import NWBHDF5IO

nwbfile_path = "/data/IBL-mesoscope-nwbfiles/full/sub-SWC054/sub-SWC054_ses-5ce2e17e-8471-42d4-8a16-21949710b328_desc-processed_behavior+ophys.nwb"

with NWBHDF5IO(nwbfile_path, "r") as io:
    nwbfile = io.read()

    # Access ROI fluorescence traces for the first FOV
    ophys = nwbfile.processing["ophys"]
    fluorescence = ophys["Fluorescence"]["RoiResponseSeriesFOV00"]
    print(fluorescence.data[:])        # shape: (n_frames, n_rois)
    print(fluorescence.timestamps[:])  # timestamps in seconds

    # Access trial table
    trials = nwbfile.trials.to_dataframe()
    print(trials.head())

    # Access wheel position
    behavior = nwbfile.processing["behavior"]
    wheel_pos = behavior["WheelPosition"]
    print(wheel_pos.data[:])

    # Access anatomical localization
    localization = nwbfile.lab_meta_data["localization"]
```

---

## Anatomical Localization Coordinate Spaces

The anatomical localization interfaces store ROI and pixel-level coordinates in two coordinate reference frames.

### IBL-Bregma space

- **Name:** `IBLBregma`
- **Origin:** Bregma (the junction of the coronal and sagittal sutures on the skull surface)
- **Units:** micrometres (µm)
- **Orientation:** RAS (Right-Anterior-Superior)
  - x = mediolateral axis, positive = right
  - y = anteroposterior axis, positive = anterior
  - z = dorsoventral axis, positive = dorsal

### Allen Common Coordinate Framework v3 (CCF v3)

- **Name:** Allen CCF v3
- **Origin:** anterior-dorsal-left corner of the reference volume
- **Units:** micrometres (µm)
- **Orientation:** PIR (Posterior-Inferior-Right) — the standard Allen CCF orientation

The `MesoscopeROIAnatomicalLocalizationInterface` stores one coordinate triplet per segmented ROI. The `MesoscopeImageAnatomicalLocalizationInterface` stores one coordinate triplet per pixel of the mean imaging plane image. Both are stored in `nwbfile.lab_meta_data["localization"]` using the `ndx-anatomical-localization` NWB extension types `AnatomicalCoordinatesTable` and `AnatomicalCoordinatesImage` respectively.

```python
with NWBHDF5IO(nwbfile_path, "r") as io:
    nwbfile = io.read()
    localization = nwbfile.lab_meta_data["localization"]

    # Per-ROI coordinates
    roi_coords = localization.AnatomicalCoordinatesTableFOV00
    ibl_bregma_xyz = roi_coords["IBLBregma"]  # shape: (n_rois, 3), units: µm
    allen_ccf_xyz  = roi_coords["AllenCCFv3"]  # shape: (n_rois, 3), units: µm

    # Per-pixel image coordinates
    image_coords = localization.AnatomicalCoordinatesImageFOV00
```
