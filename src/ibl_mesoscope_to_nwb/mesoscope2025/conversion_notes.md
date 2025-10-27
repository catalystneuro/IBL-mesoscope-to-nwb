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
- Passive video stimulus data
- Video file
- Task settings

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

