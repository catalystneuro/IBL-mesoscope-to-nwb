# Handling ScanImage "Tiled" Volume Display

## Overview

ScanImage supports a **Tiled** display mode
(`SI.hDisplay.volumeDisplayStyle == "Tiled"`) where multiple Fields of View
(FOVs) are stored within a single TIFF frame, arranged vertically with filler
pixels between them.

When ScanImage is configured with `volumeDisplayStyle = "Tiled"`:

- Each TIFF frame contains data from all FOVs stacked vertically.
- Filler pixels are inserted between each FOV tile for visual separation.
- The number of tiles equals the number of imaging ROIs in the acquisition.

For an IBL mesoscope session with 8 FOVs, one TIFF frame looks like:

```text
┌─────────────────────┐
│   FOV_00 (512 rows) │
├─────────────────────┤ ← filler pixels
│   FOV_01 (512 rows) │
├─────────────────────┤ ← filler pixels
│   FOV_02 (512 rows) │
├─────────────────────┤ ← filler pixels
│        ...          │
├─────────────────────┤ ← filler pixels
│   FOV_07 (512 rows) │
└─────────────────────┘
      512 columns
```

## Extraction strategy

`MesoscopeRawImagingExtractor`
([_meso_raw_imaging_extractor.py](../../src/ibl_mesoscope_to_nwb/mesoscope2025/datainterfaces/_meso_raw_imaging_extractor.py))
detects and handles Tiled configuration as follows:

1. **Detection** — reads `SI.hDisplay.volumeDisplayStyle` from metadata.
2. **FOV count** — reads `RoiGroups.imagingRoiGroup.rois`.
3. **Per-FOV dimensions** — uses `SI.hRoiManager.linesPerFrame` (rows) and
   `SI.hRoiManager.pixelsPerLine` (columns).
4. **Filler pixel calculation**:

   ```text
   filler_pixels = (total_frame_rows - (rows_per_FOV × num_FOVs)) / (num_FOVs - 1)
   ```

5. **FOV extraction** (via `plane_index`):

   ```text
   start_row = plane_index × (rows_per_FOV + filler_pixels)
   end_row   = start_row + rows_per_FOV
   fov_data  = full_frame[start_row:end_row, :]
   ```

## Usage

```python
from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import (
    MesoscopeRawImagingExtractor,
)

# Extract FOV_03 (plane_index=3) from tiled data
extractor = MesoscopeRawImagingExtractor(
    file_path="path/to/tiled_imaging_data.tif",
    channel_name="Channel 2",
    plane_index=3,
)

# Returns an array of shape (100, 512, 512) containing only FOV_03
data = extractor.get_series(start_sample=0, end_sample=100)
```

## Validation

The extractor validates two invariants:

- Tiles are distributed along rows, not columns.
- Filler pixels are evenly distributed (the filler count must be an integer).

This lets the extractor read individual FOVs from multi-tile TIFFs without
loading the entire frame into memory.
