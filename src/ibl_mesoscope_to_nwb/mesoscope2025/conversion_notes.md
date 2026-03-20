# Notes concerning the mesoscope2025 conversion

## Source data description

### Example Session
An example session is stored at: `F:\IBL-data-share\cortexlab\Subjects\SP061\2025-01-28\001`

### Session Structure

#### Directory Organization
```
<session_root>/
├── _ibl_experiment.description.yaml    # Session metadata and configuration
├── alf/                                # ALF (ALyx File) format processed data
│   ├── _ibl_leftCamera.dlc.pqt        # Left camera DLC tracking
│   ├── _ibl_leftCamera.features.pqt   # Left camera features
│   ├── _ibl_leftCamera.times.npy      # Left camera timestamps
│   ├── _ibl_rightCamera.dlc.pqt       # Right camera DLC tracking
│   ├── _ibl_rightCamera.features.pqt  # Right camera features
│   ├── _ibl_rightCamera.times.npy     # Right camera timestamps
│   ├── leftCamera.ROIMotionEnergy.npy # Left camera ROI motion energy
│   ├── leftROIMotionEnergy.position.npy
│   ├── licks.times.npy                # Licking event timestamps
│   ├── rightCamera.ROIMotionEnergy.npy
│   ├── rightROIMotionEnergy.position.npy
│   ├── FOV_00/ ... FOV_07/            # Field of View (FOV) specific data
│   ├── task_00/                       # Task-related data (task 0)
│   └── task_01/                       # Task-related data (task 1)
├── raw_imaging_data_00/               # Raw imaging data (acquisition 0)
│   ├── _ibl_rawImagingData.meta.json
│   ├── imaging.frames.tar.bz2         # Compressed raw imaging frames
│   └── rawImagingData.times_scanImage.npy
├── raw_imaging_data_01/               # Raw imaging data (acquisition 1)
│   ├── _ibl_rawImagingData.meta.json
│   ├── imaging.frames.tar.bz2
│   ├── rawImagingData.times_scanImage.npy
│   └── reference/                     # Reference images
├── raw_sync_data/                     # Synchronization data
│   ├── _timeline_DAQdata.meta.json
│   ├── _timeline_DAQdata.raw.npy      # Raw DAQ data
│   ├── _timeline_DAQdata.timestamps.npy
│   └── _timeline_softwareEvents.log.htsv
├── raw_task_data_00/                  # Task 0 raw data (Cued Biased Choice)
│   ├── _iblrig_encoderEvents.raw.ssv
│   ├── _iblrig_encoderPositions.raw.ssv
│   ├── _iblrig_encoderTrialInfo.raw.ssv
│   ├── _iblrig_stimPositionScreen.raw.csv
│   ├── _iblrig_syncSquareUpdate.raw.csv
│   ├── _iblrig_taskData.raw.jsonable
│   └── _iblrig_taskSettings.raw.json
├── raw_task_data_01/                  # Task 1 raw data (Passive Video)
│   ├── _iblrig_taskSettings.raw.json
│   ├── _sp_taskData.raw.pqt
│   └── _sp_video.raw.mp4
├── raw_video_data/                    # Raw camera video data
│   ├── _iblrig_leftCamera.frameData.bin
│   ├── _iblrig_leftCamera.raw.mp4
│   ├── _iblrig_rightCamera.frameData.bin
│   └── _iblrig_rightCamera.raw.mp4
└── suite2p/                           # Suite2p processing output
    ├── plane0/ ... plane7/            # One directory per imaging plane
```

### Experiment Configuration

The `_ibl_experiment.description.yaml` file contains:
- **Devices**: Camera specifications (left: 60fps 1280x1024, right: 150fps 640x512), mesoscope configuration
- **Procedures**: Imaging
- **Projects**: ibl_mesoscope_active
- **Tasks**:
  - `samuel_cuedBiasedChoiceWorld`: Active behavior task with choice paradigm
  - `_sp_passiveVideo`: Passive video presentation
- **Sync**: Timeline-based synchronization using NIDQ

### Imaging Data Organization

#### Multiple Fields of View (FOVs)
The session contains **8 Fields of View (FOV_00 to FOV_07)**, each representing a different region of interest during mesoscopic imaging.

Each FOV directory contains:

##### Processed Imaging Data
- `mpci.times.npy`: Frame timestamps for this FOV
- `mpci.badFrames.npy`: Flags for bad/corrupted frames
- `mpci.mpciFrameQC.npy`: Frame quality control metrics
- `mpciFrameQC.names.tsv`: Names/descriptions of QC metrics

##### ROI (Region of Interest) Data
- `mpci.ROIActivityF.npy`: Fluorescence traces for each ROI
- `mpci.ROIActivityDeconvolved.npy`: Deconvolved activity (spike inference)
- `mpci.ROINeuropilActivityF.npy`: Neuropil fluorescence for background subtraction
- `mpciROIs.masks.sparse_npz`: Sparse matrix of ROI masks
- `mpciROIs.neuropilMasks.sparse_npz`: Sparse matrix of neuropil masks
- `mpciROIs.cellClassifier.npy`: Cell classification results
- `mpciROIs.mpciROITypes.npy`: ROI type labels
- `mpciROIs.uuids.csv`: Unique identifiers for each ROI
- `mpciROITypes.names.tsv`: ROI type name mappings

##### Anatomical Information
- `mpciROIs.brainLocationIds_ccf_2017_estimate.npy`: CCF 2017 atlas brain region IDs
- `mpciROIs.mlapdv_estimate.npy`: ML (medial-lateral), AP (anterior-posterior), DV (dorsal-ventral) coordinates
- `mpciROIs.stackPos.npy`: Position in imaging stack
- `mpciMeanImage.images.npy`: Mean/reference images
- `mpciMeanImage.brainLocationIds_ccf_2017_estimate.npy`: Brain locations for mean image
- `mpciMeanImage.mlapdv_estimate.npy`: Coordinates for mean image

##### Suite2p Data
- `_suite2p_ROIData.raw.zip` or `_suite2p_ROIData.raw/`: Raw Suite2p processing output
- `mpciStack.timeshift.npy`: Temporal alignment corrections

#### Raw Imaging Metadata

The `_ibl_rawImagingData.meta.json` contains detailed ScanImage configuration:

**Key parameters:**
- **Imaging system**: Resonant galvo-galvo (RGG) scanner
- **Scanner frequency**: 12,018.5 Hz
- **Frame rate**: ~5.08 Hz
- **Pixel resolution**: 512 x 512 per FOV
- **Channels**: 4 channels available (green: [1,2], red: [3,4])
- **Channel saved**: Channel 2 (primary green channel)
- **Multiple ROIs**: 8 regions of interest (ROIs) per acquisition
- **Z position**: 245 µm (with FastZ enabled for field curvature correction)
- **Laser power**: 45% power fraction for imaging beam
- **Total frames**: 16,338 frames per acquisition

**FOV Details**: Each of the 8 FOVs has:
- Size: 4.6665° x 4.6665° visual angle
- Resolution: 512 x 512 pixels
- Anatomical coordinates in stereotactic space (ML, AP, DV)
- Brain region assignments (CCF 2017 atlas)

### Behavioral Data

#### Task 0: Cued Biased Choice World
Located in `raw_task_data_00/`:
- Encoder data (wheel movements)
- Trial information
- Stimulus positions
- Sync square updates for synchronization
- Task data and settings

#### Task 1: Passive Video
Located in `raw_task_data_01/`:
- Passive video stimulus data (`_sp_taskData.raw.pqt`) — per-interval start/stop timestamps
- Video file (`_sp_video.raw.mp4`) — MP4 video played as visual stimulus
- Task settings (`_iblrig_taskSettings.raw.json`) — protocol parameters and epoch timing

#### Camera Data
- **Left camera**: 60 fps, 1280x1024 resolution
- **Right camera**: 150 fps, 640x512 resolution
- DeepLabCut (DLC) tracking data available
- Motion energy calculations
- Frame timestamps

### Synchronization

Synchronization is handled via Timeline system:
- DAQ data: `_timeline_DAQdata.raw.npy`
- Timestamps: `_timeline_DAQdata.timestamps.npy`
- Software events: `_timeline_softwareEvents.log.htsv`
- Sync labels: 
  - `chrono`: Mesoscope imaging sync
  - `bpod`: Behavioral task sync
  - `audio`: Microphone/audio sync

### Suite2p Processing

The `suite2p/` directory contains processing outputs organized by imaging plane (plane0-plane7), with standard Suite2p outputs including:
- Cell detection results
- ROI masks
- Fluorescence traces
- Neuropil signals
- Cell classification

### Data Format Notes

- **NumPy arrays** (`.npy`): Numerical data, timestamps, masks
- **Parquet** (`.pqt`): Tabular data (DLC tracking, features)
- **Sparse matrices** (`.sparse_npz`): ROI and neuropil masks (memory-efficient)
- **JSON**: Metadata and configuration
- **YAML**: Experiment description
- **TSV/CSV**: Tabular metadata and labels
- **Compressed archives** (`.tar.bz2`, `.zip`): Raw imaging frames and Suite2p data

## Handling ScanImage "Tiled" Display Configuration

### Overview

ScanImage supports a "Tiled" display mode (`SI.hDisplay.volumeDisplayStyle == "Tiled"`) where multiple Fields of View (FOVs) are stored within a single TIFF frame, arranged vertically with spacing between them.

### Tiled Configuration Details

When ScanImage is configured with `volumeDisplayStyle = "Tiled"`:

- **Multiple FOVs per frame**: Each TIFF frame contains data from all FOVs stacked vertically
- **Inter-tile spacing**: Filler pixels are inserted between each FOV tile for visual separation
- **Consistent structure**: The number of tiles equals the number of imaging ROIs defined in the acquisition

### Data Structure

For the IBL mesoscope data with 8 FOVs:

```
Single TIFF frame structure (vertical arrangement):
┌─────────────────────┐
│   FOV_00 (512 rows) │
├─────────────────────┤  ← Filler pixels
│   FOV_01 (512 rows) │
├─────────────────────┤  ← Filler pixels
│   FOV_02 (512 rows) │
├─────────────────────┤  ← Filler pixels
│        ...          │
├─────────────────────┤  ← Filler pixels
│   FOV_07 (512 rows) │
└─────────────────────┘
     512 columns
```

### Extraction Implementation

The `MesoscopeRawImagingExtractor` detects and handles Tiled configuration as follows:

1. **Detection**: Checks `SI.hDisplay.volumeDisplayStyle` metadata field
2. **FOV count**: Determines number of FOVs from `RoiGroups.imagingRoiGroup.rois` metadata
3. **Dimensions**: Extracts individual FOV dimensions from ScanImage metadata:
   - `SI.hRoiManager.linesPerFrame`: Rows per FOV (512)
   - `SI.hRoiManager.pixelsPerLine`: Columns per FOV (512)
4. **Filler pixel calculation**: Computes spacing between tiles:
   ```
   filler_pixels = (total_frame_rows - (rows_per_FOV × num_FOVs)) / (num_FOVs - 1)
   ```
5. **FOV extraction**: When retrieving data for a specific FOV (via `plane_index` parameter):
   ```
   start_row = plane_index × (rows_per_FOV + filler_pixels)
   end_row = start_row + rows_per_FOV
   fov_data = full_frame[start_row:end_row, :]
   ```

### Usage

To extract data for a specific FOV from Tiled configuration files:

```python
from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import MesoscopeRawImagingExtractor

# Extract FOV_03 (plane_index=3) from tiled data
extractor = MesoscopeRawImagingExtractor(
    file_path="path/to/tiled_imaging_data.tif",
    channel_name="Channel 2",
    plane_index=3  # Extracts FOV_03
)

# Get time series data for this specific FOV
data = extractor.get_series(start_sample=0, end_sample=100)
# Returns array with shape (100, 512, 512) - only FOV_03 data
```

### Validation

The extractor performs validation to ensure data integrity:

- **Column consistency**: Verifies that tiles are distributed along rows (not columns)
- **Filler pixel consistency**: Ensures filler pixels are evenly distributed (must be integer value)

This approach allows efficient extraction of individual FOV data from multi-tile TIFF files without loading the entire dataset into memory.

## NWB Conversion Architecture

### Overview

The IBL mesoscope data is converted to NWB format through two complementary pipelines:

1. **Raw Pipeline** (`raw.py`) - Converts raw acquisition data (imaging, videos, DAQ, task settings, visual stimulus)
2. **Processed Pipeline** (`processed.py`) - Converts analyzed data (segmentation, anatomical localization, tracking, behavior, task settings)

Both pipelines produce BIDS-compliant NWB files following the naming convention:
```
sub-{subject}/sub-{subject}_ses-{eid}_desc-{raw|processed}_behavior+ophys.nwb
```

### Conversion Pipelines Comparison

| Data Stream | Raw Pipeline | Processed Pipeline | NWB Container |
|-------------|--------------|-------------------|---------------|
| **Optical Physiology** | | | |
| Raw imaging (TIFF) | ✓ | | `TwoPhotonSeries` (acquisition) |
| Motion-corrected imaging | | ✓ | `TwoPhotonSeries` (processing/ophys) |
| Segmentation (ROIs) | | ✓ | `PlaneSegmentation` (processing/ophys) |
| ROI anatomical localization (IBL-Bregma) | | ✓ | `AnatomicalCoordinatesTable` (lab_meta_data) |
| ROI anatomical localization (Allen CCF v3) | | ✓ | `AnatomicalCoordinatesTable` (lab_meta_data) |
| Mean image localization (IBL-Bregma) | | ✓ | `AnatomicalCoordinatesImage` (lab_meta_data) |
| Mean image localization (Allen CCF v3) | | ✓ | `AnatomicalCoordinatesImage` (lab_meta_data) |
| **Behavioral Tasks** | | | |
| Trials (task events) | | ✓ | `TimeIntervals["trials"]` |
| Wheel position | | ✓ | `SpatialSeries` (processing/behavior) |
| Wheel kinematics | | ✓ | `TimeSeries` (processing/behavior) |
| Wheel movements | | ✓ | `TimeIntervals` (processing/behavior) |
| Licks | | ✓ | `Events` (processing/behavior) |
| Session epochs (task settings) | ✓ | ✓ | `TimeIntervals["epochs"]` (intervals) |
| Passive intervals | | ✓ | `TimeIntervals` (intervals) |
| Passive replay stimuli | | ✓ | `TimeIntervals` (intervals) |
| Visual stimulus video | ✓ | | `OpticalSeries` (stimulus_templates) |
| Visual stimulus intervals | ✓ | | `TimeIntervals` (stimulus) |
| **Camera Data** | | | |
| Raw videos | ✓ | | `ImageSeries` (acquisition) |
| Pose estimation | | ✓ | `PoseEstimation` (processing/behavior) |
| Pupil tracking | | ✓ | `TimeSeries` (processing/behavior) |
| ROI motion energy | | ✓ | `TimeSeries` (processing/behavior) |
| **Synchronization** | | | |
| DAQ board signals | ✓ | | `TimeSeries`/`LabeledEvents` (acquisition) |

### Data Flow Diagrams

#### Raw Pipeline Data Flow

```
SOURCE DATA                          INTERFACE                              NWB CONTAINER

raw_imaging_data_XX/
├── imaging.frames/*.tif      ──→  MesoscopeRawImagingInterface      ──→  TwoPhotonSeries
└── rawImagingData.times...                                                (acquisition)

raw_sync_data/
├── _timeline_DAQdata.raw.npy  ──→  MesoscopeDAQInterface           ──→  TimeSeries (analog)
├── _timeline_DAQdata.timestamps                                          LabeledEvents (digital)
└── _timeline_DAQdata.meta.json                                           (acquisition)

raw_video_data/
├── _iblrig_leftCamera.raw.mp4  ──→  RawVideoInterface (left)      ──→  ImageSeries
├── _iblrig_rightCamera.raw.mp4 ──→  RawVideoInterface (right)     ──→  (acquisition)
└── _iblrig_bodyCamera.raw.mp4  ──→  RawVideoInterface (body)

alf/
└── _ibl_*Camera.times.npy     ──→  (timestamps for videos)

raw_task_data_XX/
├── _iblrig_taskSettings.raw.json ──→  TaskSettingsInterface       ──→  TimeIntervals["epochs"]
│                                                                        (nwbfile.epochs)
│
└── (passiveVideo collection):
    ├── _sp_video.raw.mp4         ──→  VisualStimulusInterface     ──→  OpticalSeries
    └── _sp_taskData.raw.pqt                                            (stimulus_templates)
                                                                        TimeIntervals
                                                                        (nwbfile.stimulus)
```

#### Processed Pipeline Data Flow

```
SOURCE DATA                          INTERFACE                              NWB CONTAINER

OPTICAL PHYSIOLOGY
──────────────────
suite2p/planeX/
├── imaging.frames_motion...bin ──→  MesoscopeMotionCorrected...    ──→  TwoPhotonSeries
└── (binary format)                                                       (processing/ophys)

alf/FOV_XX/
├── mpci.ROIActivityF.npy       ──→  MesoscopeSegmentation...       ──→  PlaneSegmentation
├── mpci.ROIActivityDeconv.npy                                            RoiResponseSeries
├── mpciROIs.masks.sparse_npz                                             (processing/ophys)
├── mpciMeanImage.images.npy
└── ... (ROI data)

alf/FOV_XX/
├── mpciROIs.mlapdv_estimate... ──→  MesoscopeROIAnatomical...     ──→  AnatomicalCoordinatesTable
├── mpciROIs.brainLocationIds...                                         (lab_meta_data/localization)
├── mpciMeanImage.mlapdv_est... ──→  MesoscopeImageAnatomical...   ──→  AnatomicalCoordinatesImage
└── mpciMeanImage.brainLoc...                                            (lab_meta_data/localization)

BEHAVIORAL TASKS
────────────────
_ibl_experiment.description  ──→  TaskSettingsInterface           ──→  TimeIntervals["epochs"]
raw_task_data_XX/                                                       (nwbfile.epochs)
└── _iblrig_taskSettings.raw.json

alf/task_XX/
├── _ibl_trials.*.npy           ──→  BrainwideMapTrialsInterface   ──→  TimeIntervals["trials"]
├── wheel.position.npy          ──→  MesoscopeWheelPosition...     ──→  SpatialSeries
├── wheel.velocity.npy          ──→  MesoscopeWheelKinematics...   ──→  TimeSeries
├── wheel.acceleration.npy                                               (processing/behavior)
├── wheelMoves.*.npy            ──→  MesoscopeWheelMovements...    ──→  TimeIntervals
└── licks.times.npy             ──→  LickInterface                 ──→  Events

alf/
├── passiveGabor.table.csv      ──→  PassiveReplayStimInterface    ──→  TimeIntervals (intervals)
└── passivePeriods.*.npy        ──→  PassiveIntervalsInterface     ──→  TimeIntervals (intervals)

CAMERA-BASED PROCESSING
───────────────────────
alf/
├── _ibl_*Camera.dlc.pqt        ──→  IblPoseEstimationInterface    ──→  PoseEstimation
│   (or .lightningPose.pqt)                                              (processing/behavior)
├── _ibl_*Camera.features.pqt   ──→  PupilTrackingInterface        ──→  TimeSeries (pupils)
└── *Camera.ROIMotionEnergy.npy ──→  RoiMotionEnergyInterface      ──→  TimeSeries (motion)
```

### Optical Physiology Data Streams

#### Raw Imaging (TwoPhotonSeries in acquisition)

**Source Files:**
- `raw_imaging_data_{task}/imaging.frames/*.tif` - ScanImage TIFF files
- `raw_imaging_data_{task}/rawImagingData.times_scanImage.npy` - Frame timestamps
- `raw_imaging_data_{task}/_ibl_rawImagingData.meta.json` - ScanImage metadata

**Processing:**
- Handles ScanImage "Tiled" display mode where multiple FOVs are stacked vertically in single frames
- Extracts individual FOVs using `FOV_index` parameter
- Overrides extractor timestamps with synchronized Timeline timestamps

**NWB Output:**
- One `TwoPhotonSeries` per FOV per task
- Naming: `TwoPhotonSeriesFOV{XX}Task{YY}`
- Located in: `nwbfile.acquisition`
- Linked to: `ImagingPlaneFOV{XX}` with device, grid spacing, imaging rate metadata

#### Motion-Corrected Imaging (TwoPhotonSeries in processing)

**Source Files:**
- `suite2p/plane{X}/imaging.frames_motionRegistered.bin` - Suite2p binary format
- `alf/FOV_{XX}/mpci.times.npy` - Frame timestamps
- `alf/FOV_{XX}/mpciMeanImage.images.npy` - Mean projection for frame dimensions

**Processing:**
- Memory-mapped access to binary data (int16 format)
- Shape: (num_frames, height, width)
- Validates file size against expected dimensions

**NWB Output:**
- One `TwoPhotonSeries` per FOV
- Naming: `TwoPhotonSeriesFOV{XX}`
- Located in: `nwbfile.processing["ophys"]`
- Linked to same `ImagingPlaneFOV{XX}` as raw data

#### Segmentation (PlaneSegmentation & RoiResponseSeries)

**Source Files:**
- `alf/FOV_{XX}/mpciROIs.masks.sparse_npz` - ROI pixel masks (pydata/sparse format)
- `alf/FOV_{XX}/mpciROIs.neuropilMasks.sparse_npz` - Neuropil masks (optional)
- `alf/FOV_{XX}/mpci.ROIActivityF.npy` - Fluorescence traces
- `alf/FOV_{XX}/mpci.ROIActivityDeconvolved.npy` - Deconvolved activity
- `alf/FOV_{XX}/mpci.ROIActivityFNeuropil.npy` - Neuropil traces (optional)
- `alf/FOV_{XX}/mpciROIs.cellClassifier.npy` - Cell quality scores
- `alf/FOV_{XX}/mpciROIs.mpciROITypes.npy` - Cell vs non-cell classification
- `alf/FOV_{XX}/mpciROIs.uuids.csv` - Unique ROI identifiers
- `alf/FOV_{XX}/mpciROIs.stackPos.npy` - ROI centroid positions

**Processing:**
- Converts sparse masks from pydata/sparse COO format to NWB pixel_mask format
- Separates accepted (cell) vs rejected (non-cell) ROIs
- Adds custom properties: cell_classifier, UUID

**NWB Output:**
- `PlaneSegmentation` table: `PlaneSegmentationFOV{XX}`
  - Pixel masks for each ROI
  - ROI locations (x, y, z)
  - Cell classifier scores
  - UUID identifiers
  - Background (neuropil) components
- `RoiResponseSeries`:
  - `RoiResponseSeriesRawFOV{XX}` - Fluorescence traces
  - `RoiResponseSeriesDeconvolvedFOV{XX}` - Deconvolved activity
  - `RoiResponseSeriesNeuropilFOV{XX}` - Background traces (if available)
- Summary images: Mean projection in `SegmentationImages` container
- Located in: `nwbfile.processing["ophys"]`

#### Anatomical Localization (ndx-anatomical-localization)

Anatomical localization is provided in **two coordinate spaces** for both ROIs
(`MesoscopeROIAnatomicalLocalizationInterface`) and mean projection images
(`MesoscopeImageAnatomicalLocalizationInterface`).

**Source Files for ROIs:**

- `alf/FOV_{XX}/mpciROIs.mlapdv_estimate.npy` - ML, AP, DV coordinates in IBL-Bregma space (µm)
- `alf/FOV_{XX}/mpciROIs.brainLocationIds_ccf_2017_estimate.npy` - Allen CCF 2017 region IDs

**Source Files for Mean Images:**

- `alf/FOV_{XX}/mpciMeanImage.mlapdv_estimate.npy` - Per-pixel ML/AP/DV coordinates (µm)
- `alf/FOV_{XX}/mpciMeanImage.brainLocationIds_ccf_2017_estimate.npy` - Per-pixel region IDs

**Coordinate Spaces:**

*IBL-Bregma*: RAS orientation, origin = bregma, units = µm.
  x = ML (mediolateral, +right), y = AP (anteroposterior, +anterior), z = DV (+dorsal).
  Raw ML/AP/DV values are stored directly from `mlapdv_estimate`.

*Allen CCF v3*: PIR orientation (AP, DV, ML), units = µm.
  Converted from IBL-Bregma using `iblatlas.atlas.MRITorontoAtlas(res_um=10)`.
  Conversion: divide by 1e6 (µm → m), negate AP axis (IBL +anterior → CCF +posterior),
  then call `atlas.xyz2ccf(xyz, ccf_order="apdvml")`.

**Processing:**

- Both spaces are registered in a shared `Localization` container in `nwbfile.lab_meta_data`
- Per-ROI coordinates: one row per ROI in the linked `PlaneSegmentation`
- Per-pixel coordinates (images): reshaped from (H, W, 3) → (H×W, 3) for atlas lookup,
  then reshaped back to (H, W) per axis

**NWB Output:**

- `Localization` container in `nwbfile.lab_meta_data["localization"]`
- Two registered `Space` objects: `IBLBregma` and `AllenCCFv3`
- Per-FOV `AnatomicalCoordinatesTable` objects (ROI-level, links to `PlaneSegmentationFOV{XX}`):
  - `AnatomicalCoordinatesTableIBLBregmaROIFOV{XX}` — IBL-Bregma coordinates + brain region IDs
  - `AnatomicalCoordinatesTableCCFv3ROIFOV{XX}` — Allen CCF v3 coordinates + brain region IDs
- Per-FOV `AnatomicalCoordinatesImage` objects (pixel-level, links to `MotionCorrectedTwoPhotonSeriesFOV{XX}`):
  - `AnatomicalCoordinatesImageIBLBregmaFOV{XX}` — per-pixel IBL-Bregma coordinates
  - `AnatomicalCoordinatesImageCCFv3FOV{XX}` — per-pixel Allen CCF v3 coordinates

### Behavioral Data Streams

#### Trials (TimeIntervals)

**Source Files:**
- `alf/task_{XX}/_ibl_trials.{column}.npy` - Multiple files, one per column

**Key Trial Columns:**
- Timing: `stimOn_times`, `stimOff_times`, `goCue_times`, `response_times`, `feedback_times`, `firstMovement_times`
- Stimuli: `contrastLeft`, `contrastRight`, `probabilityLeft`
- Responses: `choice` (-1=CCW, 0=no-go, 1=CW)
- Outcomes: `feedbackType` (+1=correct, -1=incorrect), `rewardVolume`
- Quality: `included` (boolean mask for quality trials)

**NWB Output:**
- `TimeIntervals` table: `nwbfile.trials`
- Each column becomes a trials table column with descriptive metadata
- Custom columns defined in `_metadata/trials.yml`

#### Wheel Behavior

**Source Files (per task):**
- `alf/task_{XX}/wheel.position.npy` - Angular position (radians)
- `alf/task_{XX}/wheel.velocity.npy` - Angular velocity (rad/s)
- `alf/task_{XX}/wheel.acceleration.npy` - Angular acceleration (rad/s²)
- `alf/task_{XX}/wheel.timestamps.npy` - Sample times
- `alf/task_{XX}/wheelMoves.intervals.npy` - Movement onset/offset times
- `alf/task_{XX}/wheelMoves.peakAmplitude.npy` - Movement amplitudes

**NWB Output:**
- `SpatialSeries`: Wheel position (processing/behavior)
  - Naming: `Task{XX}WheelPosition`
  - Description in `_metadata/wheel.yml`
- `TimeSeries`: Velocity and acceleration (processing/behavior)
  - Naming: `Task{XX}WheelVelocity`, `Task{XX}WheelAcceleration`
- `TimeIntervals`: Detected movements (processing/behavior)
  - Naming: `Task{XX}WheelMovements`
  - Columns: start_time, stop_time, peak_amplitude

#### Licks (Events)

**Source Files:**
- `alf/licks.times.npy` - Timestamps of lick events

**NWB Output:**
- `Events` object: Timestamped lick events
- Located in: `nwbfile.processing["behavior"]`

#### Session Task Settings and Epochs (TimeIntervals)

**Interface:** `TaskSettingsInterface` — used in **both raw and processed** pipelines.

**Source Files:**

- `_ibl_experiment.description` — lists all protocol names and their raw data collections
- `raw_task_data_{XX}/_iblrig_taskSettings.raw.json` — one file per protocol, contains
  `SESSION_START_TIME` and `SESSION_END_TIME` as ISO timestamps

**Processing:**

- Iterates over all protocols in `_ibl_experiment.description` using ibllib's
  `get_task_protocol()` / `get_task_collection()`
- Converts ISO timestamps to seconds relative to the NWB session start time
  (applying the lab timezone from Alyx)
- Maps protocol names to `protocol_type` and `protocol_description` via `PROTOCOLS_MAPPING`
  in `utils/tasks.py`

**NWB Output:**

- `TimeIntervals` table named `epochs` stored in `nwbfile.epochs`
- One row per protocol with columns:
  - `protocol_type` — short label (e.g. "task", "passive", "habituation")
  - `protocol_description` — human-readable description of the protocol
  - `task_settings` — full `_iblrig_taskSettings.raw.json` content as a string

#### Visual Stimulus (OpticalSeries + TimeIntervals)

**Interface:** `VisualStimulusInterface` — used in the **raw** pipeline only.

**Source Files:**

- `raw_task_data_{XX}/_sp_video.raw.mp4` — MP4 video played as visual stimulus
  during the passive protocol
- `raw_task_data_{XX}/_sp_taskData.raw.pqt` — Parquet table with per-interval
  Unix timestamps (`intervals_0`, `intervals_1`)

**Processing:**

- Automatically locates the passiveVideo collection from `_ibl_experiment.description`
  using `find_passiveVideo_collection()`
- Reads video frame rate with `cv2.VideoCapture`
- Converts Unix timestamps in `_sp_taskData` to seconds relative to the NWB session
  start time (applying the lab timezone from Alyx)

**NWB Output:**

- `OpticalSeries` named `visual_stimulus_video` in `nwbfile.stimulus_templates`
  (external reference to the MP4 file; not embedded in NWB)
- `TimeIntervals` named `visual_stimulus_intervals` in `nwbfile.stimulus`
  - Columns: `start_time`, `stop_time`, `visual_stimulus` (link to the `OpticalSeries`)

#### Session Structure — Passive Intervals and Replay Stimuli

**Passive Intervals:**

- Source: `alf/passivePeriods.*.npy`
- Output: `TimeIntervals` marking passive stimulus presentation periods

**Passive Replay Stimuli:**

- Source: `alf/passiveGabor.table.csv`
- Output: `TimeIntervals` with stimulus parameters (contrast, position, phase)

### Camera-Based Data Streams

#### Raw Videos (ImageSeries)

**Source Files (per camera):**
- `raw_video_data/_iblrig_{camera}Camera.raw.mp4` - Compressed video
- `alf/_ibl_{camera}Camera.times.npy` - Frame timestamps

**Cameras:**
- Left camera: 60 fps, 1280×1024 resolution
- Right camera: 150 fps, 640×512 resolution  
- Body camera: 60 fps, variable resolution

**NWB Output:**
- One `ImageSeries` per camera
- Located in: `nwbfile.acquisition`
- External video files (not embedded in NWB)

#### Pose Estimation (PoseEstimation)

**Source Files (per camera):**
- `alf/_ibl_{camera}Camera.dlc.pqt` - DeepLabCut tracking (fallback)
- `alf/_ibl_{camera}Camera.lightningPose.pqt` - Lightning Pose tracking (preferred)

**Tracked Body Parts:**
- Varies by camera view (eyes, nose, paws, tongue, etc.)
- Each part has x, y coordinates and likelihood score

**NWB Output:**
- `PoseEstimation` container (ndx-pose extension)
- Located in: `nwbfile.processing["behavior"]`
- One per camera with all body parts

#### Pupil Tracking (TimeSeries)

**Source Files (left/right cameras only):**
- `alf/_ibl_{camera}Camera.features.pqt` - Pupil diameter estimates

**Measurements:**
- `pupilDiameter_raw` - Raw diameter estimates
- `pupilDiameter_smooth` - Smoothed and interpolated diameter

**NWB Output:**
- `TimeSeries`: `RawPupilDiameter`, `SmoothedPupilDiameter`
- Located in: `nwbfile.processing["behavior"]`
- Metadata defined in `_metadata/pupils.yml`

#### ROI Motion Energy (TimeSeries)

**Source Files (per camera):**
- `alf/{camera}Camera.ROIMotionEnergy.npy` - Motion energy time series
- `alf/{camera}ROIMotionEnergy.position.npy` - ROI position/size

**NWB Output:**
- `TimeSeries` per camera
- Captures motion within defined ROI
- Located in: `nwbfile.processing["behavior"]`

### DAQ Synchronization Data Streams

#### Timeline DAQ Board

**Source Files:**
- `raw_sync_data/_timeline_DAQdata.raw.npy` - Multi-channel analog recordings
- `raw_sync_data/_timeline_DAQdata.timestamps.npy` - Start/end times
- `raw_sync_data/_timeline_DAQdata.meta.json` - Channel wiring configuration

**Channel Types:**

**Analog Channels** (continuous signals):
- `chrono` - Mesoscope frame sync (imaging timestamps)
- `photoDiode` - Screen photodiode (visual stimulus timing)
- `GalvoX`, `GalvoY` - Galvo mirror positions
- `RemoteFocus1`, `RemoteFocus2` - Z-focus control
- `LaserPower` - Laser power modulation
- `reward_valve` - Reward delivery valve signal

**Digital Channels** (event detection):
- `neural_frames` - Imaging frame TTL pulses
- `volume_counter` - Volume/plane counter
- `bpod` - Behavioral task controller sync
- `frame2ttl` - Screen refresh sync
- `left_camera`, `right_camera`, `body_camera` - Camera frame triggers
- `audio` - Audio/microphone sync
- `rotary_encoder` - Wheel encoder pulses

**Processing:**
- Channel mapping defined dynamically from `_timeline_DAQdata.meta.json`
- Digital channels converted to `LabeledEvents` with polarity labels (0/1 → low/high)
- Analog channels stored as `TimeSeries`

**NWB Output:**
- Analog: `TimeSeries` per device in `nwbfile.acquisition`
- Digital: `LabeledEvents` (ndx-events extension) per device
- Metadata defined in `_metadata/mesoscope_DAQ_metadata.yaml`
- Device info: NIDAQ model, sample rate, channel specifications

### Data Format Notes

**File Formats Used:**
- **TIFF** - Raw imaging data (multi-page ScanImage format)
- **Binary (.bin)** - Motion-corrected imaging (memory-mapped int16)
- **Sparse matrices** - ROI masks (pydata/sparse format, not scipy.sparse)
- **Parquet (.pqt)** - Tabular data (DLC, features)
- **NumPy (.npy)** - Numerical arrays, timestamps
- **JSON/YAML** - Metadata and configuration
- **MP4** - Compressed video files

**Special Handling:**
- ScanImage Tiled mode: Multiple FOVs per TIFF frame
- Sparse masks: COO format → NWB pixel_mask conversion
- Timeline sync: Session-wide synchronization across all devices
- External videos: Referenced but not embedded in NWB
