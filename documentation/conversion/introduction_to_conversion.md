# NWB Conversion

This section documents how IBL mesoscope sessions are converted to NWB.

## Architecture

IBL-mesoscope-to-nwb uses [NeuroConv](https://neuroconv.readthedocs.io/), which
organizes conversions around two concepts:

- **DataInterface** — a reader for a single data modality (e.g. raw imaging,
  segmentation, DAQ, wheel, pose estimation). Each interface knows which ONE
  API files it needs, how to download them, and how to add its data to an
  `NWBFile`.
- **NWBConverter** — orchestrates multiple interfaces into a single
  conversion, handling metadata merging and temporal alignment across
  modalities.

```text
IBL session (ONE API)
    │
    ├──► MesoscopeRawImagingInterface ─────┐
    ├──► MesoscopeDAQInterface ────────────┤
    ├──► MesoscopeSegmentationInterface ───┤
    ├──► MesoscopeROIAnatomicalLoc. ───────┼──► NWBConverter ──► NWB file
    ├──► BrainwideMapTrialsInterface ──────┤
    ├──► IblPoseEstimationInterface ───────┤
    └──► … (one interface per modality) ───┘
```

Source code lives under [src/ibl_mesoscope_to_nwb/mesoscope2025/](../../src/ibl_mesoscope_to_nwb/mesoscope2025/):

- [datainterfaces/](../../src/ibl_mesoscope_to_nwb/mesoscope2025/datainterfaces/) —
  mesoscope-specific interfaces (raw imaging, motion-corrected imaging,
  segmentation, anatomical localization, DAQ, wheel, visual stimulus, task
  settings).
- [conversion/](../../src/ibl_mesoscope_to_nwb/mesoscope2025/conversion/) —
  pipeline entry points (`raw.py`, `processed.py`, `download.py`).
- [nwbconverter.py](../../src/ibl_mesoscope_to_nwb/mesoscope2025/nwbconverter.py) —
  the converter class that combines interfaces.
- [convert_session.py](../../src/ibl_mesoscope_to_nwb/mesoscope2025/convert_session.py) —
  single-session entry point.
- [convert_all_sessions.py](../../src/ibl_mesoscope_to_nwb/mesoscope2025/convert_all_sessions.py) —
  parallel batch conversion wrapper.
- [_metadata/](../../src/ibl_mesoscope_to_nwb/mesoscope2025/_metadata/) — YAML
  metadata templates (trials columns, wheel, pupils, DAQ wiring).

## Two conversion pipelines

Each session is converted through two independent pipelines, producing two
separate BIDS-compliant NWB files:

| Pipeline | Entry point | Output filename |
|----------|-------------|-----------------|
| Raw | `convert_raw_session` | `sub-{subject}_ses-{eid}_desc-raw_behavior+ophys.nwb` |
| Processed | `convert_processed_session` | `sub-{subject}_ses-{eid}_desc-processed_behavior+ophys.nwb` |

Output files are written to:

```text
{output_path}/{full|stub}/sub-{subject}/sub-{subject}_ses-{eid}_desc-{raw|processed}_behavior+ophys.nwb
```

`full` is used for standard conversions; `stub` is used when `stub_test=True`.

### Raw pipeline

Converts data as acquired, without post-processing. See [conversion_modalities.md](conversion_modalities.md)
for the full table of NWB locations.

| Interface | Data |
|-----------|------|
| `MesoscopeRawImagingInterface` | ScanImage TIFF files, one per FOV per task |
| `MesoscopeDAQInterface` | Timeline DAQ board analog and digital channels |
| `TaskSettingsInterface` | Session epoch timing from `_iblrig_taskSettings.raw.json` |
| `VisualStimulusInterface` | Passive visual stimulus video and presentation intervals |
| `RawVideoInterface` | Raw behavioral camera videos (left, right, body) |

### Processed pipeline

Converts analyzed outputs (Suite2p segmentation, motion-corrected imaging,
anatomical localization, behavioral features).

| Interface | Data |
|-----------|------|
| `MesoscopeMotionCorrectedImagingInterface` | Motion-corrected binary imaging, one per FOV |
| `MesoscopeSegmentationInterface` | ROI masks, fluorescence traces, deconvolved traces |
| `MesoscopeROIAnatomicalLocalizationInterface` | Per-ROI coordinates in IBL-Bregma and Allen CCF v3 |
| `MesoscopeImageAnatomicalLocalizationInterface` | Per-pixel mean-image coordinates |
| `TaskSettingsInterface` | Session epoch timing |
| `MesoscopeWheelPositionInterface` | Wheel position per task |
| `MesoscopeWheelKinematicsInterface` | Wheel velocity / acceleration per task |
| `MesoscopeWheelMovementsInterface` | Detected wheel movements per task |
| `BrainwideMapTrialsInterface` | Trial table |
| `IblPoseEstimationInterface` | Pose estimation (Lightning Pose / DLC) per camera |
| `PupilTrackingInterface` | Pupil diameter tracking per camera |
| `RoiMotionEnergyInterface` | ROI motion energy per camera |

## Documents in this section

- [conversion_overview.md](conversion_overview.md) — how to run conversions
  (Python API, stub mode, parallel, availability checks, downloads).
- [conversion_modalities.md](conversion_modalities.md) — where each modality
  lives inside the NWB file.
- [anatomical_localization.md](anatomical_localization.md) — IBL-Bregma vs.
  Allen CCF v3 coordinate spaces.
- [source_data.md](source_data.md) — on-disk session layout the pipeline
  reads from.
- [scanimage_tiled_mode.md](scanimage_tiled_mode.md) — handling ScanImage's
  "Tiled" volume display mode.
