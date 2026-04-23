# Tutorials

Four Jupyter notebooks walk through loading and inspecting a converted NWB
file. They live under
[src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/](../src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/).

| Notebook | Pipeline | Topics covered |
|----------|----------|----------------|
| [raw.ipynb](../src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/raw.ipynb) | Raw | ScanImage TIFF imaging, DAQ analog/digital channels, raw behavioural videos, visual stimulus video and intervals |
| [processed.ipynb](../src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/processed.ipynb) | Processed | Motion-corrected imaging, ROI segmentation (masks, fluorescence, deconvolved traces), behavioural processing module |
| [behavior.ipynb](../src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/behavior.ipynb) | Processed | Trial table, wheel position / kinematics / movements, pose estimation, pupil tracking, ROI motion energy |
| [anatomical_localization.ipynb](../src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/anatomical_localization.ipynb) | Processed | ROI and per-pixel coordinates in IBL-Bregma and Allen CCF v3 spaces |

## Running the notebooks

Each notebook opens an existing NWB file produced by the conversion pipeline
and demonstrates the relevant access patterns. You will need an NWB file on
disk or a DANDI asset URL. Start Jupyter from the repository root:

```bash
jupyter lab src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/
```

A small utility module,
[load_nwb_utils.py](../src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/load_nwb_utils.py),
provides shared helpers used across the notebooks (e.g., DANDI streaming
helpers and `NWBHDF5IO` wrappers).

## See also

- [conversion/conversion_overview.md](conversion/conversion_overview.md) —
  how to produce the NWB files the notebooks consume.
- [conversion/conversion_modalities.md](conversion/conversion_modalities.md) —
  reference table of where each modality lives inside the NWB file.
- [conversion/anatomical_localization.md](conversion/anatomical_localization.md) —
  details of the two coordinate frames used by the anatomical-localization
  notebook.
