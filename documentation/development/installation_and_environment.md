# Installation and Environment

## Python version

This project targets **Python ≥ 3.10**. The reference environment used to
produce the NWB files is Python **3.13**.

## Quick installation

From a clone of the repository:

```bash
git clone https://github.com/catalystneuro/IBL-mesoscope-to-nwb.git
cd IBL-mesoscope-to-nwb
pip install -e .
```

### With conda

A minimal conda environment file is provided
([make_env.yml](../../make_env.yml)). It installs Python and pip, then runs
`pip install -e .`:

```bash
conda env create --file make_env.yml
conda activate ibl-mesoscope-to-nwb-env
```

## Declared dependencies

The top-level dependencies declared in
[pyproject.toml](../../pyproject.toml) are:

| Package | Purpose |
|---------|---------|
| `neuroconv` | Conversion framework (data interfaces, converters) |
| `nwbinspector` | Validation of produced NWB files |
| `roiextractors` | Imaging / segmentation extractors (ScanImage, Suite2p) |
| `ONE-api` | IBL ONE API client for data access |
| `ibllib` | IBL toolbox (task protocols, DAQ loaders) |
| `iblatlas` | IBL atlas transforms (IBL-Bregma ↔ Allen CCF v3) |
| `pynwb` | NWB file reading / writing |
| `sparse` | pydata/sparse COO arrays for ROI masks |
| `tqdm` | Progress bars for batch conversion |
| `opencv-python-headless` | Video frame reading (visual stimulus MP4) |
| `python-dateutil` | Timezone handling during session metadata parsing |
| `ndx-anatomical-localization` | NWB extension for anatomical coordinates |
| `ndx-ibl-bwm` (git) | Brain-Wide Map NWB types reused for trials |
| `ibl-to-nwb` (git) | Base interfaces shared with the Brain-Wide Map pipeline |
| `ndx-ibl` (git) | `IblSubject` and other IBL-specific NWB types |

Transitive dependencies (e.g. `ndx-events`, `ndx-pose`, `pynwb`, `numpy`,
`pandas`) are pulled in automatically.

## Reference environment snapshot

The exact package versions used to produce the reference NWB files are
captured in [requirements_freeze.txt](../../requirements_freeze.txt) at the
repository root. This is a `pip freeze` of the conda env
`ibl-mesoscope-to-nwb-env` that produced the conversions — a human-readable
reference, not intended for direct installation.

Key pinned versions from that snapshot:

| Package | Version |
|---------|---------|
| Python | 3.13 |
| pynwb | 3.1.3 |
| hdmf | 4.3.1 |
| neuroconv | 0.9.0 |
| roiextractors | 0.7.3 |
| nwbinspector | 0.6.5 |
| ONE-api | 3.4.1 |
| ibllib | 3.4.3 |
| iblatlas | 0.10.0 |
| ibl_to_nwb | 0.3.0 |
| ndx-ibl | 0.3.0 |
| ndx-ibl-bwm | 0.1.0 |
| ndx-anatomical-localization | 0.1.0 |
| ndx-events | 0.2.1 |
| ndx-pose | 0.2.2 |
| sparse | 0.17.0 |
| numpy | 2.2.0 |
| pandas | 2.3.3 |
| dandi | 0.74.1 |

See [lock_files.md](lock_files.md) for instructions on recreating the exact
environment.

## Regenerating `requirements_freeze.txt`

After changing dependencies, regenerate the freeze snapshot from inside the
active environment:

```bash
pip freeze > requirements_freeze.txt
```

or from outside the environment:

```bash
/path/to/python -m pip freeze > requirements_freeze.txt
```
