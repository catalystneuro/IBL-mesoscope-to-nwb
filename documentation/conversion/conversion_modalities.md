# Conversion Modalities

This document summarises where each source-data modality ends up inside the
produced NWB files. For a deep dive into design decisions for individual
modalities (source file paths, processing, naming conventions) see the
[conversion notes](../../src/ibl_mesoscope_to_nwb/mesoscope2025/conversion_notes.md)
that accompany the source code.

## NWB locations per modality

| Data | NWB type | NWB location | Pipeline |
|------|----------|--------------|----------|
| Raw imaging (per FOV per task) | `TwoPhotonSeries` | `nwbfile.acquisition` | Raw |
| DAQ analog channels | `TimeSeries` | `nwbfile.acquisition` | Raw |
| DAQ digital channels | `LabeledEvents` (ndx-events) | `nwbfile.acquisition` | Raw |
| Raw behavioral videos | `ImageSeries` | `nwbfile.acquisition` | Raw |
| Visual stimulus video | `OpticalSeries` | `nwbfile.stimulus_templates` | Raw |
| Visual stimulus intervals | `TimeIntervals` | `nwbfile.stimulus` | Raw |
| Session/task epochs | `TimeIntervals` named `"epochs"` | `nwbfile.epochs` | Raw + Processed |
| Motion-corrected imaging (per FOV) | `TwoPhotonSeries` | `nwbfile.processing["ophys"]` | Processed |
| ROI segmentation (per FOV) | `PlaneSegmentation` + `RoiResponseSeries` | `nwbfile.processing["ophys"]` | Processed |
| Anatomical coords (ROI-level) | `AnatomicalCoordinatesTable` | `nwbfile.lab_meta_data["localization"]` | Processed |
| Anatomical coords (image-level) | `AnatomicalCoordinatesImage` | `nwbfile.lab_meta_data["localization"]` | Processed |
| Trials | `TimeIntervals` | `nwbfile.trials` | Processed |
| Wheel position | `SpatialSeries` | `nwbfile.processing["behavior"]` | Processed |
| Wheel kinematics (velocity, acceleration) | `TimeSeries` | `nwbfile.processing["behavior"]` | Processed |
| Wheel movements | `TimeIntervals` | `nwbfile.processing["behavior"]` | Processed |
| Licks | `Events` (ndx-events) | `nwbfile.processing["behavior"]` | Processed |
| Pose estimation (per camera) | `PoseEstimation` (ndx-pose) | `nwbfile.processing["behavior"]` | Processed |
| Pupil tracking (per camera) | `TimeSeries` | `nwbfile.processing["behavior"]` | Processed |
| ROI motion energy (per camera) | `TimeSeries` | `nwbfile.processing["behavior"]` | Processed |

## Naming conventions

Per-FOV objects are suffixed with the zero-padded FOV index:

- Raw imaging: `TwoPhotonSeriesFOV{XX}Task{YY}`
- Motion-corrected imaging: `TwoPhotonSeriesFOV{XX}`
- Segmentation table: `PlaneSegmentationFOV{XX}`
- ROI response series: `RoiResponseSeriesRawFOV{XX}`,
  `RoiResponseSeriesDeconvolvedFOV{XX}`,
  `RoiResponseSeriesNeuropilFOV{XX}`
- ROI anatomical coordinates: `AnatomicalCoordinatesTableIBLBregmaROIFOV{XX}`,
  `AnatomicalCoordinatesTableCCFv3ROIFOV{XX}`
- Image anatomical coordinates: `AnatomicalCoordinatesImageIBLBregmaFOV{XX}`,
  `AnatomicalCoordinatesImageCCFv3FOV{XX}`

Per-task wheel objects are prefixed with the task index:

- `Task{XX}WheelPosition`, `Task{XX}WheelVelocity`,
  `Task{XX}WheelAcceleration`, `Task{XX}WheelMovements`.

## NWB extensions used

- [ndx-anatomical-localization](https://pypi.org/project/ndx-anatomical-localization/) —
  `AnatomicalCoordinatesTable`, `AnatomicalCoordinatesImage`, `Localization`
  container in `lab_meta_data`.
- [ndx-events](https://pypi.org/project/ndx-events/) — `LabeledEvents` for DAQ
  digital channels, `Events` for licks.
- [ndx-pose](https://pypi.org/project/ndx-pose/) — `PoseEstimation` for
  DeepLabCut and Lightning Pose outputs.
- [ndx-ibl](https://github.com/catalystneuro/ndx-ibl) — `IblSubject` and other
  IBL-specific metadata types.
- [ndx-ibl-bwm](https://github.com/int-brain-lab/ndx-ibl-bwm) — Brain-Wide Map
  specific types reused for trial metadata.
