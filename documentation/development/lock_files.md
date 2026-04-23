# Reproducing the Conversion Environment

For reproducibility, the repository ships two environment artifacts:

| File | Tool | Purpose |
|------|------|---------|
| [`make_env.yml`](../../make_env.yml) | conda | Create a fresh env, install the package editable |
| [`requirements_freeze.txt`](../../requirements_freeze.txt) | pip / human | Exact package list from the reference env |

## Fresh install with conda

```bash
conda env create --file make_env.yml
conda activate ibl-mesoscope-to-nwb-env
```

This creates a new conda environment named `ibl-mesoscope-to-nwb-env`, installs
Python 3.13 and pip, then installs the project in editable mode. Transitive
dependencies are resolved by pip at install time, so versions may drift from
the reference environment over time.

## Reproducing exact versions

`requirements_freeze.txt` is a `pip freeze` snapshot of the Python environment
used to produce the reference NWB files. It is the simplest way to inspect
what was installed (one package/version per line) and can be used to recreate
the same environment:

```bash
conda create -n ibl-mesoscope-to-nwb-env python=3.13 pip
conda activate ibl-mesoscope-to-nwb-env
pip install -r requirements_freeze.txt
pip install -e .
```

Caveats:

- `requirements_freeze.txt` does not contain hashes and is platform-specific —
  Windows wheels for packages with native extensions (e.g. `imagecodecs`,
  `opencv-python-headless`, `PyQt5`) may not be installable on macOS or Linux
  without adjustments.
- Git-sourced packages (`ibl_to_nwb`, `ndx-ibl`, `ndx-ibl-bwm`) appear with
  their installed versions but the commit hash that produced them is not
  captured automatically. When reproducing, pin these via the git URLs in
  [`pyproject.toml`](../../pyproject.toml) rather than the freeze file.

## Regenerating `requirements_freeze.txt`

From inside the active environment:

```bash
pip freeze > requirements_freeze.txt
```
