# Source Data

This document describes the on-disk layout of an IBL mesoscope session — the
input the conversion pipeline reads from. A full version of these notes,
including extractor-level design decisions, lives alongside the source code at
[conversion_notes.md](../../src/ibl_mesoscope_to_nwb/mesoscope2025/conversion_notes.md).

## Example session

An example session is stored at:

```text
F:\IBL-data-share\cortexlab\Subjects\SP061\2025-01-28\001
```

## Directory layout

```text
<session_root>/
├── _ibl_experiment.description.yaml   # Session metadata and configuration
├── alf/                               # ALF (ALyx File) processed data
│   ├── _ibl_leftCamera.dlc.pqt        # Left camera DLC tracking
│   ├── _ibl_leftCamera.features.pqt   # Left camera features (pupil, etc.)
│   ├── _ibl_leftCamera.times.npy      # Left camera timestamps
│   ├── _ibl_rightCamera.*             # Analogous right-camera files
│   ├── leftCamera.ROIMotionEnergy.npy
│   ├── leftROIMotionEnergy.position.npy
│   ├── rightCamera.ROIMotionEnergy.npy
│   ├── rightROIMotionEnergy.position.npy
│   ├── licks.times.npy                # Licking event timestamps
│   ├── FOV_00/ … FOV_07/              # One directory per Field of View
│   ├── task_00/                       # Task-related data (task 0)
│   └── task_01/                       # Task-related data (task 1)
├── raw_imaging_data_00/               # Raw imaging data (acquisition 0)
│   ├── _ibl_rawImagingData.meta.json
│   ├── imaging.frames.tar.bz2         # Compressed raw imaging frames
│   └── rawImagingData.times_scanImage.npy
├── raw_imaging_data_01/               # Raw imaging data (acquisition 1)
│   └── …
├── raw_sync_data/                     # Synchronization data
│   ├── _timeline_DAQdata.meta.json
│   ├── _timeline_DAQdata.raw.npy
│   ├── _timeline_DAQdata.timestamps.npy
│   └── _timeline_softwareEvents.log.htsv
├── raw_task_data_00/                  # Task 0 (Cued Biased Choice World)
│   ├── _iblrig_encoderEvents.raw.ssv
│   ├── _iblrig_encoderPositions.raw.ssv
│   ├── _iblrig_encoderTrialInfo.raw.ssv
│   ├── _iblrig_stimPositionScreen.raw.csv
│   ├── _iblrig_syncSquareUpdate.raw.csv
│   ├── _iblrig_taskData.raw.jsonable
│   └── _iblrig_taskSettings.raw.json
├── raw_task_data_01/                  # Task 1 (Passive Video)
│   ├── _iblrig_taskSettings.raw.json
│   ├── _sp_taskData.raw.pqt
│   └── _sp_video.raw.mp4
├── raw_video_data/                    # Raw camera video
│   ├── _iblrig_leftCamera.raw.mp4
│   ├── _iblrig_rightCamera.raw.mp4
│   └── _iblrig_bodyCamera.raw.mp4
└── suite2p/                           # Suite2p output (one directory per plane)
    └── plane0/ … plane7/
```

## Experiment configuration

The `_ibl_experiment.description.yaml` file contains:

- **Devices** — camera specifications (left 60 fps 1280×1024, right 150 fps
  640×512), mesoscope configuration.
- **Procedures** — `Imaging`.
- **Projects** — `ibl_mesoscope_active`.
- **Tasks** — typically `samuel_cuedBiasedChoiceWorld` (active choice) and
  `_sp_passiveVideo` (passive replay).
- **Sync** — Timeline-based synchronization via NIDQ.

## Imaging data

### Fields of view (FOVs)

A typical session contains **8 FOVs** (`FOV_00` … `FOV_07`), each representing a
different region of interest during mesoscopic imaging.

Each FOV directory contains:

**Processed imaging data**

- `mpci.times.npy` — frame timestamps.
- `mpci.badFrames.npy` — bad-frame mask.
- `mpci.mpciFrameQC.npy` — per-frame QC metrics.
- `mpciFrameQC.names.tsv` — QC metric names.

**ROI data**

- `mpci.ROIActivityF.npy` — fluorescence traces per ROI.
- `mpci.ROIActivityDeconvolved.npy` — deconvolved activity (spike inference).
- `mpci.ROINeuropilActivityF.npy` — neuropil fluorescence.
- `mpciROIs.masks.sparse_npz` — sparse ROI masks (pydata/sparse COO format).
- `mpciROIs.neuropilMasks.sparse_npz` — sparse neuropil masks.
- `mpciROIs.cellClassifier.npy` — cell classification scores.
- `mpciROIs.mpciROITypes.npy` — ROI type labels.
- `mpciROIs.uuids.csv` — unique ROI identifiers.
- `mpciROITypes.names.tsv` — ROI type name mappings.

**Anatomical information**

- `mpciROIs.brainLocationIds_ccf_2017_estimate.npy` — Allen CCF 2017 region
  IDs per ROI.
- `mpciROIs.mlapdv_estimate.npy` — IBL-Bregma ML/AP/DV coordinates per ROI.
- `mpciROIs.stackPos.npy` — stack position.
- `mpciMeanImage.images.npy` — mean/reference images.
- `mpciMeanImage.brainLocationIds_ccf_2017_estimate.npy` — per-pixel region IDs.
- `mpciMeanImage.mlapdv_estimate.npy` — per-pixel IBL-Bregma coordinates.

### Raw imaging metadata

`_ibl_rawImagingData.meta.json` contains the full ScanImage configuration:

| Parameter | Typical value |
|-----------|---------------|
| Imaging system | Resonant galvo-galvo (RGG) scanner |
| Scanner frequency | 12 018.5 Hz |
| Frame rate | ~5.08 Hz |
| Pixel resolution (per FOV) | 512 × 512 |
| Channels available | 4 (green: [1, 2], red: [3, 4]) |
| Channel saved | Channel 2 (primary green) |
| Imaging ROIs per acquisition | 8 |
| Z position | 245 µm (FastZ for field-curvature correction) |
| Laser power | 45 % |
| Frames per acquisition | 16 338 |

Each FOV has a size of **4.6665° × 4.6665°** visual angle, 512 × 512 pixels,
with anatomical ML/AP/DV coordinates and CCF 2017 region assignments.

## Behavioural data

### Task 0 — Cued Biased Choice World

Located in `raw_task_data_00/`. Contains wheel encoder data, trial info,
stimulus positions, sync-square updates, and task settings.

### Task 1 — Passive Video

Located in `raw_task_data_01/`. Contains the MP4 stimulus (`_sp_video.raw.mp4`),
per-interval start/stop timestamps (`_sp_taskData.raw.pqt`), and task settings.

### Camera data

- **Left camera** — 60 fps, 1280 × 1024, DLC/Lightning Pose tracking available.
- **Right camera** — 150 fps, 640 × 512, DLC/Lightning Pose tracking available.
- **Body camera** — 60 fps, variable resolution.

## Synchronization

Timeline-based synchronization via NIDQ:

- `raw_sync_data/_timeline_DAQdata.raw.npy` — multichannel analog recording.
- `raw_sync_data/_timeline_DAQdata.timestamps.npy` — start/end times.
- `raw_sync_data/_timeline_DAQdata.meta.json` — channel wiring.
- `raw_sync_data/_timeline_softwareEvents.log.htsv` — software events log.

Key sync labels: `chrono` (mesoscope imaging), `bpod` (behavioural task),
`audio` (microphone), `neural_frames`, `frame2ttl`, camera frame triggers,
`rotary_encoder`.

## Suite2p output

`suite2p/` is organised by imaging plane (`plane0` … `plane7`) with standard
Suite2p outputs: cell detection, ROI masks, fluorescence traces, neuropil
signals, cell classification.

## File-format cheatsheet

| Extension | Purpose |
|-----------|---------|
| `.npy` | Numerical arrays, timestamps, masks |
| `.pqt` | Tabular data (DLC, Lightning Pose, features) |
| `.sparse_npz` | Sparse ROI / neuropil masks (pydata/sparse COO) |
| `.json` | Metadata and configuration |
| `.yaml` | Experiment description |
| `.tsv` / `.csv` | Tabular metadata and labels |
| `.tar.bz2` / `.zip` | Compressed raw imaging frames, Suite2p bundles |
| `.mp4` | Compressed video (behavioural cameras, passive stimulus) |
| `.bin` | Motion-corrected imaging (memory-mapped int16) |
