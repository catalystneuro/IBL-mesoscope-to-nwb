# IBL-mesoscope-to-nwb

NWB conversion scripts for IBL two-photon mesoscope sessions (ScanImage +
Suite2p + behaviour) to the
[Neurodata Without Borders](https://nwb-overview.readthedocs.io/) data format.

> **Full documentation lives under [`documentation/`](documentation/introduction_to_documentation.md).**
> This README is a short overview; everything below is covered in more detail
> there.

## Using the data

The conversion produces two BIDS-compliant NWB files per session — one for raw
acquisition data, one for processed/analysed data. Load a converted file with
[PyNWB](https://pynwb.readthedocs.io/):

```python
from pynwb import NWBHDF5IO

with NWBHDF5IO(nwbfile_path, "r") as io:
    nwbfile = io.read()

    # ROI fluorescence traces for the first FOV
    fluorescence = nwbfile.processing["ophys"]["Fluorescence"]["RoiResponseSeriesFOV00"]

    # Trial table
    trials = nwbfile.trials.to_dataframe()

    # Anatomical coordinates (IBL-Bregma + Allen CCF v3)
    localization = nwbfile.lab_meta_data["localization"]
```

## Tutorials

Four Jupyter notebooks walk through a converted NWB file. See the
[tutorials index](documentation/tutorials.md) for full descriptions.

| Notebook | Description |
|----------|-------------|
| [raw.ipynb](src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/raw.ipynb) | Raw pipeline: ScanImage TIFF, DAQ, behavioural videos |
| [processed.ipynb](src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/processed.ipynb) | Processed pipeline: motion-corrected imaging, segmentation |
| [behavior.ipynb](src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/behavior.ipynb) | Trials, wheel, pose estimation, pupil tracking |
| [anatomical_localization.ipynb](src/ibl_mesoscope_to_nwb/mesoscope2025/tutorials/anatomical_localization.ipynb) | ROI and pixel-level coordinates (IBL-Bregma, Allen CCF v3) |

## Running conversions

### Installation

```bash
git clone https://github.com/catalystneuro/IBL-mesoscope-to-nwb
cd IBL-mesoscope-to-nwb
pip install -e .
```

Or with conda:

```bash
conda env create --file make_env.yml
conda activate ibl-mesoscope-to-nwb-env
```

See [installation_and_environment.md](documentation/development/installation_and_environment.md)
for dependency details and [lock_files.md](documentation/development/lock_files.md)
for reproducing the exact reference environment.

### Convert a single session

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

convert_raw_session(eid=eid, one=one, output_path=output_path, verbose=True)
convert_processed_session(eid=eid, one=one, output_path=output_path, verbose=True)
```

See [conversion_overview.md](documentation/conversion/conversion_overview.md)
for stub-test mode, parallel conversion, availability checks, and
pre-downloading data.

## Pipelines

Each session produces two independent NWB files:

| Pipeline | Function | Output |
|----------|----------|--------|
| Raw | `convert_raw_session` | `…_desc-raw_behavior+ophys.nwb` |
| Processed | `convert_processed_session` | `…_desc-processed_behavior+ophys.nwb` |

Each pipeline is a composition of NeuroConv
[DataInterfaces](https://neuroconv.readthedocs.io/en/main/user_guide/datainterfaces.html)
combined by an
[NWBConverter](https://neuroconv.readthedocs.io/en/main/user_guide/nwbconverter.html).
See [introduction_to_conversion.md](documentation/conversion/introduction_to_conversion.md)
for the list of interfaces per pipeline and
[conversion_modalities.md](documentation/conversion/conversion_modalities.md)
for where each modality lands inside the NWB file.

## Repository structure

```text
IBL-mesoscope-to-nwb/
├── LICENSE
├── README.md
├── pyproject.toml
├── make_env.yml
├── requirements_freeze.txt                 # pip freeze of the reference env
├── documentation/                          # Full documentation
│   ├── introduction_to_documentation.md
│   ├── tutorials.md
│   ├── conversion/
│   │   ├── introduction_to_conversion.md
│   │   ├── conversion_overview.md
│   │   ├── conversion_modalities.md
│   │   ├── anatomical_localization.md
│   │   ├── source_data.md
│   │   └── scanimage_tiled_mode.md
│   └── development/
│       ├── installation_and_environment.md
│       └── lock_files.md
└── src/
    └── ibl_mesoscope_to_nwb/
        └── mesoscope2025/
            ├── conversion_notes.md         # In-code conversion design notes
            ├── tutorials/                  # Jupyter notebooks
            ├── conversion/                 # raw.py, processed.py, download.py
            ├── datainterfaces/             # Mesoscope-specific interfaces
            ├── nwbconverter.py
            ├── convert_session.py          # Single-session entry point
            ├── convert_all_sessions.py     # Parallel batch conversion
            └── _metadata/                  # YAML metadata templates
```

## Helpful definitions

- A [DataInterface](https://neuroconv.readthedocs.io/en/main/user_guide/datainterfaces.html)
  converts a single data modality to NWB from a distinct set of files.
- An [NWBConverter](https://neuroconv.readthedocs.io/en/main/user_guide/nwbconverter.html)
  combines multiple interfaces, specifying temporal alignment between
  modalities.
- Conversion scripts decide which sessions to convert, instantiate the
  NWBConverter, and write NWB files to the output directory.
