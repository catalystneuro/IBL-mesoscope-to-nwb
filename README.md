# IBL-mesoscope-to-nwb

NWB conversion scripts for IBL mesoscope data to the
[Neurodata Without Borders](https://nwb-overview.readthedocs.io/) data format.

## Documentation

- [USAGE.md](src/ibl_mesoscope_to_nwb/mesoscope2025/USAGE.md) — full usage guide: pipelines, downloading, stub testing, parallel conversion, NWB file structure
- [conversion_notes.md](src/ibl_mesoscope_to_nwb/mesoscope2025/conversion_notes.md) — source data description, interface details, and conversion design decisions

### Tutorials (Jupyter notebooks)

| Notebook | Description |
|----------|-------------|
| [raw.ipynb](src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/raw.ipynb) | Raw pipeline: ScanImage TIFF, DAQ, behavioral videos |
| [processed.ipynb](src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/processed.ipynb) | Processed pipeline: motion-corrected imaging, segmentation, behavior |
| [behavior.ipynb](src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/behavior.ipynb) | Behavioral data: trials, wheel, pose estimation, pupil tracking |
| [anatomical_localization.ipynb](src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/anatomical_localization.ipynb) | ROI and pixel-level anatomical coordinates (IBL-Bregma, Allen CCF v3) |

---

## Installation

### From PyPI

```bash
pip install IBL-mesoscope-to-nwb
```

### From GitHub (recommended for development)

```bash
git clone https://github.com/catalystneuro/IBL-mesoscope-to-nwb
cd IBL-mesoscope-to-nwb
pip install -e .
```

We recommend installing inside a virtual environment. Using conda:

```bash
conda env create --file make_env.yml
conda activate ibl-mesoscope-to-nwb-env
```

---

## Quick Start

```python
from pathlib import Path
from one.api import ONE
from ibl_mesoscope_to_nwb.mesoscope2025.conversion import convert_raw_session, convert_processed_session

eid = "5ce2e17e-8471-42d4-8a16-21949710b328"
one = ONE()
output_path = Path("/data/IBL-mesoscope-nwbfiles")

# Convert raw acquisition data (ScanImage TIFFs, DAQ, videos)
result = convert_raw_session(eid=eid, one=one, output_path=output_path, verbose=True)
print(f"Raw NWB: {result['nwbfile_path']} ({result['nwb_size_gb']:.2f} GB)")

# Convert processed/analyzed data (motion correction, segmentation, behavior)
result = convert_processed_session(eid=eid, one=one, output_path=output_path, verbose=True)
print(f"Processed NWB: {result['nwbfile_path']}")
```

See [USAGE.md](src/ibl_mesoscope_to_nwb/mesoscope2025/USAGE.md) for full details.

---

## Conversion Pipeline

There are two independent pipelines, each producing a separate NWB file:

| Pipeline | Function | Output |
|----------|----------|--------|
| Raw | `convert_raw_session` | `..._desc-raw_behavior+ophys.nwb` |
| Processed | `convert_processed_session` | `..._desc-processed_behavior+ophys.nwb` |

### Raw pipeline data interfaces

| Interface | Data |
|-----------|------|
| `MesoscopeRawImagingInterface` | ScanImage TIFF files, one per FOV |
| `MesoscopeDAQInterface` | Timeline DAQ board analog and digital channels |
| `TaskSettingsInterface` | Session epoch timing |
| `VisualStimulusInterface` | Passive visual stimulus video and presentation intervals |
| `RawVideoInterface` | Raw behavioral camera videos (left, right, body) |

### Processed pipeline data interfaces

| Interface | Data |
|-----------|------|
| `MesoscopeMotionCorrectedImagingInterface` | Motion-corrected binary imaging, one per FOV |
| `MesoscopeSegmentationInterface` | ROI masks, fluorescence traces, deconvolved traces |
| `MesoscopeROIAnatomicalLocalizationInterface` | Per-ROI coordinates in IBL-Bregma and Allen CCF v3 |
| `MesoscopeImageAnatomicalLocalizationInterface` | Per-pixel mean image coordinates |
| `TaskSettingsInterface` | Session epoch timing |
| `MesoscopeWheelPositionInterface` | Wheel position per task |
| `MesoscopeWheelKinematicsInterface` | Wheel velocity/speed per task |
| `MesoscopeWheelMovementsInterface` | Detected wheel movements per task |
| `BrainwideMapTrialsInterface` | Trial table |
| `IblPoseEstimationInterface` | Pose estimation (Lightning Pose / DLC) per camera |
| `PupilTrackingInterface` | Pupil diameter tracking per camera |
| `RoiMotionEnergyInterface` | ROI motion energy per camera |

---

## Repository Structure

```text
IBL-mesoscope-to-nwb/
├── LICENSE
├── make_env.yml
├── pyproject.toml
├── README.md
└── src/
    └── ibl_mesoscope_to_nwb/
        └── mesoscope2025/
            ├── USAGE.md                  # Usage guide
            ├── conversion_notes.md       # Design notes and source data description
            ├── tutorials/                # Jupyter notebooks
            │   ├── raw.ipynb
            │   ├── processed.ipynb
            │   ├── behavior.ipynb
            │   └── anatomical_localization.ipynb
            ├── conversion/               # Conversion functions
            │   ├── convert_raw.py
            │   └── convert_processed.py
            ├── datainterfaces/           # Data interface classes
            ├── nwbconverter.py           # NWBConverter class
            ├── convert_session.py        # Single-session entry point
            ├── convert_all_sessions.py   # Batch/parallel conversion
            ├── metadata.yml              # Experiment-level metadata
            └── __init__.py
```

---

## Helpful Definitions

A [DataInterface](https://neuroconv.readthedocs.io/en/main/user_guide/datainterfaces.html) converts a single data modality to NWB from a distinct set of files.

An [NWBConverter](https://neuroconv.readthedocs.io/en/main/user_guide/nwbconverter.html) combines multiple data interfaces, specifying temporal alignment between modalities.

The conversion scripts determine which sessions to convert, instantiate the NWBConverter, and write NWB files to the output directory.
