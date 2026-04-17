# Anatomical Localization

The processed pipeline stores every segmented ROI and every pixel of each
mean-imaging plane in two coordinate reference frames: **IBL-Bregma** and
**Allen CCF v3**. Both are registered under `nwbfile.lab_meta_data["localization"]`
using the [ndx-anatomical-localization](https://pypi.org/project/ndx-anatomical-localization/)
extension.

## IBL-Bregma space

- **Name:** `IBLBregma`
- **Origin:** Bregma (junction of coronal and sagittal sutures on the skull)
- **Units:** micrometres (µm)
- **Orientation:** RAS (Right–Anterior–Superior)
  - `x` = mediolateral, positive = right
  - `y` = anteroposterior, positive = anterior
  - `z` = dorsoventral, positive = dorsal

Raw `ML / AP / DV` values are stored directly from `mpciROIs.mlapdv_estimate`
and `mpciMeanImage.mlapdv_estimate`.

## Allen Common Coordinate Framework v3 (CCF v3)

- **Name:** `AllenCCFv3`
- **Origin:** anterior–dorsal–left corner of the reference volume
- **Units:** micrometres (µm)
- **Orientation:** PIR (Posterior–Inferior–Right), the standard Allen CCF
  orientation.

Conversion from IBL-Bregma is performed with
`iblatlas.atlas.MRITorontoAtlas(res_um=10)`: divide by `1e6` to go µm → m,
negate the AP axis (IBL `+anterior` → CCF `+posterior`), then call
`atlas.xyz2ccf(xyz, ccf_order="apdvml")`.

## NWB layout

Two interfaces populate the `Localization` container:

- `MesoscopeROIAnatomicalLocalizationInterface` — one coordinate triplet per
  segmented ROI, stored in `AnatomicalCoordinatesTable` linked to the per-FOV
  `PlaneSegmentation`.
- `MesoscopeImageAnatomicalLocalizationInterface` — one coordinate triplet per
  pixel of the mean imaging plane image, stored in
  `AnatomicalCoordinatesImage` linked to the per-FOV motion-corrected
  `TwoPhotonSeries`.

### Per-FOV objects

| Object | Contents |
|--------|----------|
| `AnatomicalCoordinatesTableIBLBregmaROIFOV{XX}` | IBL-Bregma ROI coordinates + CCF 2017 region IDs |
| `AnatomicalCoordinatesTableCCFv3ROIFOV{XX}` | Allen CCF v3 ROI coordinates + CCF 2017 region IDs |
| `AnatomicalCoordinatesImageIBLBregmaFOV{XX}` | Per-pixel IBL-Bregma coordinates |
| `AnatomicalCoordinatesImageCCFv3FOV{XX}` | Per-pixel Allen CCF v3 coordinates |

### Accessing coordinates

```python
from pynwb import NWBHDF5IO

with NWBHDF5IO(nwbfile_path, "r") as io:
    nwbfile = io.read()
    localization = nwbfile.lab_meta_data["localization"]

    # Per-ROI coordinates
    roi_bregma = localization.AnatomicalCoordinatesTableIBLBregmaROIFOV00
    ibl_bregma_xyz = roi_bregma[:]       # (n_rois, 3) in µm

    roi_ccf = localization.AnatomicalCoordinatesTableCCFv3ROIFOV00
    allen_ccf_xyz = roi_ccf[:]            # (n_rois, 3) in µm

    # Per-pixel image coordinates
    image_bregma = localization.AnatomicalCoordinatesImageIBLBregmaFOV00
```

See the [anatomical_localization.ipynb](../tutorials.md) tutorial for a
visualisation of both spaces.
