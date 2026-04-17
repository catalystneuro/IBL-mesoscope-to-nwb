# IBL-mesoscope-to-nwb Documentation

This directory contains the full documentation for the IBL-mesoscope-to-nwb
conversion pipeline. The pipeline converts International Brain Laboratory (IBL)
two-photon mesoscope sessions — acquired with ScanImage on a resonant
galvo-galvo scanner — into [Neurodata Without Borders](https://nwb-overview.readthedocs.io/)
(NWB) files, using [NeuroConv](https://neuroconv.readthedocs.io/) and
[ROIExtractors](https://roiextractors.readthedocs.io/).

The documentation is organized into three sections:

## 1. Conversion

How to run conversions, what the pipeline does, and how each data modality is
mapped to NWB.

- [introduction_to_conversion.md](conversion/introduction_to_conversion.md) —
  overview of the two conversion pipelines (raw and processed) and the
  architecture (interfaces + NWBConverter).
- [conversion_overview.md](conversion/conversion_overview.md) — how to run a
  conversion: Python API, stub test mode, parallel batch conversion, checking
  data availability, pre-downloading data.
- [conversion_modalities.md](conversion/conversion_modalities.md) — where each
  data modality (imaging, segmentation, anatomical localization, wheel, trials,
  pose, pupil, DAQ, visual stimulus, videos) lands inside the NWB file.
- [anatomical_localization.md](conversion/anatomical_localization.md) — the two
  coordinate reference frames (IBL-Bregma and Allen CCF v3) and how ROI-level
  and image-level coordinates are stored.
- [source_data.md](conversion/source_data.md) — description of the on-disk
  IBL mesoscope session layout (ALF, raw_imaging_data, suite2p, etc.) that the
  conversion reads from.
- [scanimage_tiled_mode.md](conversion/scanimage_tiled_mode.md) — how the
  pipeline handles ScanImage's "Tiled" volume display mode, where multiple FOVs
  are stacked vertically in a single TIFF frame.

## 2. Tutorials

- [tutorials.md](tutorials.md) — index of the four Jupyter notebooks that walk
  through inspecting a converted NWB file: raw imaging, processed imaging,
  behavior, and anatomical localization.

## 3. Development

- [installation_and_environment.md](development/installation_and_environment.md) —
  how to install the package (pip, conda) and the exact environment used for
  the reference conversion (`requirements_freeze.txt`).
- [lock_files.md](development/lock_files.md) — reproducing the conversion
  environment from the pinned dependency snapshot.
