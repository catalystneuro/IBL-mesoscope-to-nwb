# IBL Mesoscope Raw Imaging Metadata Guide

## Overview

The `_ibl_rawImagingData.meta.json` file contains comprehensive metadata about the mesoscopic imaging acquisition, including ScanImage configuration, ROI definitions, laser settings, and coordinate transformations.

**File location:** `<session_root>/raw_imaging_data_00/_ibl_rawImagingData.meta.json`

## Loading the Metadata

```python
import json
from pathlib import Path

# Load the metadata file
metadata_path = Path("F:/IBL-data-share/cortexlab/Subjects/SP061/2025-01-28/001/raw_imaging_data_00/_ibl_rawImagingData.meta.json")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
```

---

## Top-Level Structure

### Version
```python
version = metadata['version']  # "0.2.0"
```

The metadata format version.

---

## Channel Configuration

### Channel IDs
```python
channel_ids = metadata['channelID']
# {
#     "green": [1, 2],      # Green fluorescence channels
#     "red": [3, 4],        # Red fluorescence channels
#     "primary": [2, 4],    # Primary channels for each color
#     "secondary": [1, 3]   # Secondary channels for each color
# }

# Access specific channels
green_channels = channel_ids['green']        # [1, 2]
red_channels = channel_ids['red']            # [3, 4]
primary_green = channel_ids['primary'][0]    # 2
primary_red = channel_ids['primary'][1]      # 4
```

**Explanation:** 
- Defines which physical PMT channels correspond to green and red fluorescence
- Primary channels are typically used for the main signals
- Channel 2 is the primary green channel that was saved during acquisition

### Channel Saved
```python
channel_saved = metadata['channelSaved']  # 2
```

The channel number that was actually recorded during the session.

---

## Laser Power Calibration

```python
laser_cal = metadata['laserPowerCalibration']

# Voltage values (V)
voltages = laser_cal['V']  # [0, 0.05, 0.1, ..., 5]

# Percentage values (%)
percentages = laser_cal['Prcnt']  # [0, 1, 2, ..., 100]

# Power in milliwatts (mW)
power_mw = laser_cal['mW']  # [0, 12, 24, ..., 1200]

# Dual beam voltage
dual_voltage = laser_cal['dualV']  # [0, 0.02, 0.04, ..., 2]

# Dual beam percentage
dual_percent = laser_cal['dualPrc']  # [0, 1, 2, ..., 100]

# Example: Find power for a specific voltage
voltage_of_interest = 2.5
idx = voltages.index(voltage_of_interest)
power_at_voltage = power_mw[idx]  # 600 mW
```

**Explanation:** 
- Lookup table for converting between laser voltage, percentage, and actual power
- Used to calibrate laser intensity during imaging
- Maximum power: 1200 mW at 5V or 100%

---

## Image Orientation and Coordinate Systems

### Image Orientation
```python
image_orientation = metadata['imageOrientation']

# Direction of positive medial-lateral (ML) in image coordinates
positive_ml = image_orientation['positiveML']  # [0, -1]

# Direction of positive anterior-posterior (AP) in image coordinates
positive_ap = image_orientation['positiveAP']  # [-1, 0]
```

**Explanation:** 
- Defines how brain coordinates map to image pixel coordinates
- `[0, -1]` means positive ML is in the negative Y direction
- `[-1, 0]` means positive AP is in the negative X direction

### Center Position (Degrees)
```python
center_deg = metadata['centerDeg']
x_deg = center_deg['x']  # 0
y_deg = center_deg['y']  # 0
```

Center of the imaging field in degrees of visual angle.

### Center Position (Millimeters)
```python
center_mm = metadata['centerMM']

# Image center in mm
x_mm = center_mm['x']    # 0
y_mm = center_mm['y']    # 0

# Stereotactic coordinates
ml_mm = center_mm['ML']  # 2.6 mm (medial-lateral)
ap_mm = center_mm['AP']  # -2.0 mm (anterior-posterior)
```

**Explanation:** 
- Stereotactic coordinates relative to bregma
- ML: 2.6 mm lateral
- AP: -2.0 mm posterior to bregma

### Coordinate Transformation
```python
coords_tf = metadata['coordsTF']
# [[0, -0.15],      # Rotation/scaling matrix
#  [-0.15, 0],      # Rotation/scaling matrix
#  [2.6, -2]]       # Translation vector [ML, AP]
```

Transformation matrix to convert from image coordinates to brain coordinates.

---

## Acquisition Timing

```python
# Acquisition start time
acq_start = metadata['acquisitionStartTime']
# [2025, 1, 28, 10, 49, 53.448]
# [year, month, day, hour, minute, second]

# Total number of frames
n_frames = metadata['nFrames']  # 16338
```

---

## PMT Configuration

```python
pmt_gain = metadata['PMTGain']  # [] (empty - not used in this setup)
```

Photomultiplier tube gain settings (if applicable).

---

## ScanImage Parameters

### Overview
```python
si_params = metadata['scanImageParams']
```

### Objective and Resolution
```python
objective_resolution = si_params['objectiveResolution']  # 150 (lines per mm)
```

### Scanner Configuration (hScan2D)
```python
scan2d = si_params['hScan2D']

# Flyto time between scanfields
flyto_time = scan2d['flytoTimePerScanfield']  # 0.003 seconds

# Resonant scanner frequency
scanner_freq = scan2d['scannerFrequency']  # 12018.5 Hz
```

**Explanation:**
- `flytoTimePerScanfield`: Dead time when scanner moves between ROIs
- `scannerFrequency`: Resonant mirror oscillation frequency

### FastZ Configuration (hFastZ)
```python
fast_z = si_params['hFastZ']

# Field curvature correction enabled
field_curve_corr = fast_z['enableFieldCurveCorr']  # 1 (True)

# Current Z position
z_position = fast_z['position']  # 245 μm
```

**Explanation:**
- FastZ allows rapid z-axis movement
- Field curvature correction compensates for optical aberrations
- Position indicates imaging depth

### ROI Manager (hRoiManager)
```python
roi_manager = si_params['hRoiManager']

# Frame timing
scan_frame_period = roi_manager['scanFramePeriod']  # 0.19703 seconds
scan_frame_rate = roi_manager['scanFrameRate']      # 5.07538 Hz
scan_volume_rate = roi_manager['scanVolumeRate']    # 5.07538 Hz

# Line timing
line_period = roi_manager['linePeriod']  # 4.16025e-05 seconds (~24,035 lines/sec)
```

**Explanation:**
- Frame period: Time to complete one full frame across all ROIs
- Frame rate: ~5.08 frames per second
- Volume rate: For 3D imaging (same as frame rate in 2D mode)
- Line period: Time to scan one horizontal line

### Stack Manager (hStackManager)
```python
stack_manager = si_params['hStackManager']

# Number of z-slices
num_slices = stack_manager['numSlices']  # 245

# Current z positions
zs = stack_manager['zs']  # 245
zs_relative = stack_manager['zsRelative']      # [245, 245]
zs_all_actuators = stack_manager['zsAllActuators']  # [245, 245]
```

**Explanation:**
- Defines z-stack parameters if volumetric imaging is used
- All arrays show imaging at 245 μm depth

---

## Field of View (FOV) Definitions

### Overview
```python
fov_list = metadata['FOV']
n_fovs = len(fov_list)  # 8 FOVs
```

### Accessing Individual FOV Data

```python
# Get first FOV
fov_0 = fov_list[0]

# Basic information
slice_id = fov_0['slice_id']          # 0
roi_uuid = fov_0['roiUUID']           # "CCE276EFA6244E74"
z_position = fov_0['Zs']              # 245 μm

# Image dimensions
nx, ny, nz = fov_0['nXnYnZ']          # [512, 512, 1]

# Line indices in the full frame
line_indices = fov_0['lineIdx']       # [1, 2, 3, ..., 512]
```

### Spatial Coordinates (Degrees)
```python
deg_coords = fov_0['Deg']

# Corner positions in degrees
top_left_deg = deg_coords['topLeft']          # [-12.940583, -2.898583]
top_right_deg = deg_coords['topRight']        # [-8.274083, -2.898583]
bottom_left_deg = deg_coords['bottomLeft']    # [-12.940583, 1.767917]
bottom_right_deg = deg_coords['bottomRight']  # [-8.274083, 1.767917]
```

### Spatial Coordinates (Millimeters)
```python
mm_coords = fov_0['MM']

# Corner positions in mm (relative to center)
top_left_mm = mm_coords['topLeft']          # [3.034787, -0.058913]
top_right_mm = mm_coords['topRight']        # [3.034787, -0.758888]
bottom_left_mm = mm_coords['bottomLeft']    # [2.334812, -0.058913]
bottom_right_mm = mm_coords['bottomRight']  # [2.334812, -0.758888]
```

### Stereotactic Coordinates (MLAPDV)
```python
mlapdv = fov_0['MLAPDV']

# Each corner has [ML, AP, DV] coordinates in μm
top_left_stereo = mlapdv['topLeft']
# [2888.423, -100.538, -847.471]
# ML: 2888 μm lateral, AP: -100 μm posterior, DV: -847 μm ventral

top_right_stereo = mlapdv['topRight']
# [2946.682, -786.718, -644.179]

bottom_left_stereo = mlapdv['bottomLeft']
# [2247.423, -78.424, -565.523]

bottom_right_stereo = mlapdv['bottomRight']
# [2302.040, -765.333, -371.515]

center_stereo = mlapdv['center']
# [2603.152, -430.815, -591.225]
```

**Explanation:**
- ML (medial-lateral): Distance from midline
- AP (anterior-posterior): Distance from bregma
- DV (dorsal-ventral): Depth from brain surface
- All values in micrometers (μm)

### Brain Region Mapping
```python
brain_locations = fov_0['brainLocationIds']

# Allen CCF 2017 atlas IDs for each corner
top_left_id = brain_locations['topLeft']          # 450
top_right_id = brain_locations['topRight']        # 981
bottom_left_id = brain_locations['bottomLeft']    # 450
bottom_right_id = brain_locations['bottomRight']  # 450
center_id = brain_locations['center']             # 450
```

**Explanation:**
- Brain location IDs from Allen Common Coordinate Framework (CCF) 2017
- Used to identify which brain regions are being imaged
- Example IDs:
  - 450: Primary visual cortex (VISp)
  - 981: Retrosplenial cortex
  - 1030: Secondary visual cortex

---

## Processing All FOVs

### Extract Key Information from All FOVs
```python
for i, fov in enumerate(metadata['FOV']):
    print(f"\nFOV {i}:")
    print(f"  UUID: {fov['roiUUID']}")
    print(f"  Dimensions: {fov['nXnYnZ']}")
    print(f"  Z position: {fov['Zs']} μm")
    print(f"  Center (ML, AP, DV): {fov['MLAPDV']['center']}")
    print(f"  Center brain region ID: {fov['brainLocationIds']['center']}")
    print(f"  Number of lines: {len(fov['lineIdx'])}")
```

### Create FOV Summary Table
```python
import pandas as pd

fov_data = []
for i, fov in enumerate(metadata['FOV']):
    fov_data.append({
        'FOV_ID': i,
        'UUID': fov['roiUUID'],
        'Z_um': fov['Zs'],
        'Width_px': fov['nXnYnZ'][0],
        'Height_px': fov['nXnYnZ'][1],
        'ML_um': fov['MLAPDV']['center'][0],
        'AP_um': fov['MLAPDV']['center'][1],
        'DV_um': fov['MLAPDV']['center'][2],
        'Brain_Region_ID': fov['brainLocationIds']['center'],
        'Start_Line': fov['lineIdx'][0],
        'End_Line': fov['lineIdx'][-1]
    })

df_fovs = pd.DataFrame(fov_data)
print(df_fovs)
```

### Calculate FOV Sizes
```python
import numpy as np

for i, fov in enumerate(metadata['FOV']):
    # Get corner coordinates
    mlapdv = fov['MLAPDV']
    
    # Calculate width and height in μm
    width_um = np.linalg.norm(
        np.array(mlapdv['topRight']) - np.array(mlapdv['topLeft'])
    )
    height_um = np.linalg.norm(
        np.array(mlapdv['bottomLeft']) - np.array(mlapdv['topLeft'])
    )
    
    print(f"FOV {i}: {width_um:.1f} μm × {height_um:.1f} μm")
```

---

## Raw ScanImage Metadata

The `rawScanImageMeta` field contains extensive ScanImage-specific configuration. Access key fields:

### Artist Metadata (ROI Groups)
```python
artist = metadata['rawScanImageMeta']['Artist']
roi_groups = artist['RoiGroups']

# Imaging ROI group
imaging_group = roi_groups['imagingRoiGroup']
group_name = imaging_group['name']  # "SP065_8rois_new"
group_uuid = imaging_group['roiUuid']  # "6AD203BF3D6D875B"

# Get all ROIs in the group
rois = imaging_group['rois']
n_rois = len(rois)  # 8 ROIs

# Access first ROI
roi_0 = rois[0]
roi_name = roi_0['name']  # "ROI 8"
roi_uuid = roi_0['roiUuid']  # "CCE276EFA6244E74"
roi_z = roi_0['zs']  # 120 (z position in ScanImage units)
```

### Scanfield Properties
```python
# Get scanfield for first ROI
scanfield = rois[0]['scanfields']

# Center position in degrees
center_xy = scanfield['centerXY']  # [-10.607333, -0.565333]

# Size in degrees
size_xy = scanfield['sizeXY']  # [4.6665, 4.6665]

# Rotation
rotation = scanfield['rotationDegrees']  # 0

# Pixel resolution
pixel_res = scanfield['pixelResolutionXY']  # [512, 512]

# Affine transformation matrix
affine = scanfield['affine']
# [[4.6665, 0, -12.940583],
#  [0, 4.6665, -2.898583],
#  [0, 0, 1]]
```

### ImageDescription (Frame Metadata)
```python
image_desc = metadata['rawScanImageMeta']['ImageDescription']

# Parse the string (it's a formatted text block)
# Contains per-frame information like:
# - frameNumbers
# - frameTimestamps_sec
# - acquisitionNumbers
# - epoch (timestamp)
```

### Software Configuration
```python
software = metadata['rawScanImageMeta']['Software']

# This is a very long string containing all ScanImage configuration
# Key parameters are already extracted in scanImageParams
# Access specific values by searching the string:
if 'SI.VERSION_MAJOR = 2020' in software:
    print("ScanImage 2020 version")
```

---

## Complete Example: Extract All Metadata

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

def load_imaging_metadata(session_path):
    """
    Load and parse imaging metadata from IBL mesoscope session.
    
    Parameters
    ----------
    session_path : str or Path
        Path to session directory
        
    Returns
    -------
    dict
        Dictionary containing organized metadata
    """
    session_path = Path(session_path)
    metadata_file = session_path / "raw_imaging_data_00" / "_ibl_rawImagingData.meta.json"
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Organize key information
    organized = {
        'version': metadata['version'],
        'n_frames': metadata['nFrames'],
        'start_time': metadata['acquisitionStartTime'],
        'channel_saved': metadata['channelSaved'],
        
        # Timing
        'frame_rate': metadata['scanImageParams']['hRoiManager']['scanFrameRate'],
        'frame_period': metadata['scanImageParams']['hRoiManager']['scanFramePeriod'],
        'line_period': metadata['scanImageParams']['hRoiManager']['linePeriod'],
        
        # Scanner
        'scanner_frequency': metadata['scanImageParams']['hScan2D']['scannerFrequency'],
        'flyto_time': metadata['scanImageParams']['hScan2D']['flytoTimePerScanfield'],
        
        # Z position
        'z_position': metadata['scanImageParams']['hFastZ']['position'],
        'field_curve_corr': metadata['scanImageParams']['hFastZ']['enableFieldCurveCorr'],
        
        # Coordinates
        'center_ml_mm': metadata['centerMM']['ML'],
        'center_ap_mm': metadata['centerMM']['AP'],
        
        # FOVs
        'n_fovs': len(metadata['FOV']),
        'fovs': []
    }
    
    # Extract FOV information
    for i, fov in enumerate(metadata['FOV']):
        fov_info = {
            'id': i,
            'uuid': fov['roiUUID'],
            'z_um': fov['Zs'],
            'dimensions': fov['nXnYnZ'],
            'center_mlapdv': fov['MLAPDV']['center'],
            'brain_region_id': fov['brainLocationIds']['center'],
            'line_range': (fov['lineIdx'][0], fov['lineIdx'][-1]),
            'n_lines': len(fov['lineIdx'])
        }
        organized['fovs'].append(fov_info)
    
    return organized

# Usage
session_path = "F:/IBL-data-share/cortexlab/Subjects/SP061/2025-01-28/001"
metadata = load_imaging_metadata(session_path)

print(f"Frame rate: {metadata['frame_rate']:.2f} Hz")
print(f"Number of FOVs: {metadata['n_fovs']}")
print(f"Total frames: {metadata['n_frames']}")

# Create summary DataFrame
fov_df = pd.DataFrame(metadata['fovs'])
print("\nFOV Summary:")
print(fov_df)
```

---

## Useful Helper Functions

### Convert Laser Voltage to Power
```python
def voltage_to_power(voltage, metadata):
    """Convert laser voltage to power in mW."""
    laser_cal = metadata['laserPowerCalibration']
    voltages = laser_cal['V']
    powers = laser_cal['mW']
    
    # Find closest voltage
    idx = min(range(len(voltages)), key=lambda i: abs(voltages[i] - voltage))
    return powers[idx]

# Example
power = voltage_to_power(2.5, metadata)  # Returns 600 mW
```

### Get FOV by UUID
```python
def get_fov_by_uuid(uuid, metadata):
    """Get FOV data by its UUID."""
    for fov in metadata['FOV']:
        if fov['roiUUID'] == uuid:
            return fov
    return None

# Example
fov = get_fov_by_uuid("CCE276EFA6244E74", metadata)
```

### Calculate FOV Coverage
```python
def get_fov_coverage(metadata):
    """Calculate spatial coverage of all FOVs."""
    all_ml = []
    all_ap = []
    all_dv = []
    
    for fov in metadata['FOV']:
        mlapdv = fov['MLAPDV']
        for corner in ['topLeft', 'topRight', 'bottomLeft', 'bottomRight']:
            ml, ap, dv = mlapdv[corner]
            all_ml.append(ml)
            all_ap.append(ap)
            all_dv.append(dv)
    
    coverage = {
        'ml_range': (min(all_ml), max(all_ml)),
        'ap_range': (min(all_ap), max(all_ap)),
        'dv_range': (min(all_dv), max(all_dv))
    }
    
    return coverage

# Example
coverage = get_fov_coverage(metadata)
print(f"ML range: {coverage['ml_range'][0]:.1f} to {coverage['ml_range'][1]:.1f} μm")
print(f"AP range: {coverage['ap_range'][0]:.1f} to {coverage['ap_range'][1]:.1f} μm")
print(f"DV range: {coverage['dv_range'][0]:.1f} to {coverage['dv_range'][1]:.1f} μm")
```

---

## Summary of Key Parameters

| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| Frame Rate | 5.08 | Hz | Imaging frequency |
| Frame Period | 0.197 | s | Time per full frame |
| Scanner Frequency | 12,018.5 | Hz | Resonant mirror frequency |
| Number of FOVs | 8 | - | Fields of view |
| FOV Size | 512 × 512 | pixels | Resolution per FOV |
| Total Frames | 16,338 | - | Frames acquired |
| Z Position | 245 | μm | Imaging depth |
| Channel Saved | 2 | - | Green primary channel |
| Laser Power (max) | 1,200 | mW | Maximum available power |
| Line Period | 41.6 | μs | Time per scan line |
| Flyto Time | 3.0 | ms | Dead time between FOVs |

---

## Notes for NWB Conversion

1. **Frame timestamps**: Use `acquisitionStartTime` combined with frame indices and `scanFramePeriod`
2. **FOV organization**: Each FOV can be stored as a separate `ImagingPlane` in NWB
3. **Spatial metadata**: Store stereotactic coordinates (ML, AP, DV) in imaging plane metadata
4. **Brain regions**: Include CCF atlas IDs for anatomical reference
5. **Line indices**: Critical for reconstructing multi-ROI data from raw files
6. **Laser power**: Store calibration curve for reproducibility
7. **Coordinate transforms**: Use `coordsTF` matrix for aligning to brain atlas

---

## Example: Creating NWB ImagingPlane Objects

Below is a complete example showing how to populate NWB `ImagingPlane` fields using the metadata from this JSON file.

### Complete Code Example

```python
import numpy as np
from pynwb import NWBFile
from pynwb.ophys import ImagingPlane, OpticalChannel
from pynwb.device import Device
from datetime import datetime
import json
from pathlib import Path

# Load metadata
metadata_path = Path("F:/IBL-data-share/cortexlab/Subjects/SP061/2025-01-28/001/raw_imaging_data_00/_ibl_rawImagingData.meta.json")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Create NWB file (example)
session_start_time = datetime(*metadata['acquisitionStartTime'][:6])
nwbfile = NWBFile(
    session_description='IBL mesoscope imaging session',
    identifier='SP061_2025-01-28_001',
    session_start_time=session_start_time
)

# Create Device
device = Device(
    name='Mesoscope',
    description='Two-photon mesoscope with resonant scanning',
    manufacturer='Custom IBL mesoscope'
)
nwbfile.add_device(device)

# Create OpticalChannel
optical_channel = OpticalChannel(
    name='GreenChannel',
    description='Green fluorescence channel (primary)',
    emission_lambda=510.0  # GCaMP emission wavelength in nm
)

# Process each FOV as a separate ImagingPlane
for fov_idx, fov in enumerate(metadata['FOV']):
    
    # Extract FOV-specific data
    fov_uuid = fov['roiUUID']
    center_mlapdv = fov['MLAPDV']['center']  # [ML, AP, DV] in micrometers
    brain_region_id = fov['brainLocationIds']['center']
    dimensions = fov['nXnYnZ']  # [width, height, depth] in pixels
    
    # Calculate grid spacing (pixel size in micrometers, then convert to meters)
    mlapdv = fov['MLAPDV']
    top_left = np.array(mlapdv['topLeft'])
    top_right = np.array(mlapdv['topRight'])
    bottom_left = np.array(mlapdv['bottomLeft'])
    
    # Width and height in micrometers
    width_um = np.linalg.norm(top_right - top_left)
    height_um = np.linalg.norm(bottom_left - top_left)
    
    # Pixel size in micrometers
    pixel_size_x = width_um / dimensions[0]  # μm/pixel
    pixel_size_y = height_um / dimensions[1]  # μm/pixel
    
    # Convert to meters for NWB
    grid_spacing = [
        pixel_size_x * 1e-6,  # x spacing in meters
        pixel_size_y * 1e-6   # y spacing in meters
    ]
    
    # Origin coordinates (top-left corner in stereotactic space)
    # Convert from micrometers to meters
    origin_coords = [
        top_left[0] * 1e-6,  # ML in meters
        top_left[1] * 1e-6,  # AP in meters  
        top_left[2] * 1e-6   # DV in meters
    ]
    
    # Create ImagingPlane
    imaging_plane = ImagingPlane(
        name=f'ImagingPlaneFOV{fov_idx:02d}',
        
        optical_channel=[optical_channel],
        
        description=(
            f'Field of view {fov_idx} (UUID: {fov_uuid}). '
            f'Center location: ML={center_mlapdv[0]:.1f}μm, '
            f'AP={center_mlapdv[1]:.1f}μm, DV={center_mlapdv[2]:.1f}μm. '
            f'Allen CCF 2017 brain region ID: {brain_region_id}. '
            f'Image dimensions: {dimensions[0]}x{dimensions[1]} pixels.'
        ),
        
        device=device,
        
        excitation_lambda=920.0,  # Two-photon excitation wavelength in nm (typical for GCaMP)
        
        indicator='GCaMP6s',  # Or appropriate calcium indicator used
        
        location=(
            f'Brain region ID {brain_region_id} (Allen CCF 2017). '
            f'Stereotactic coordinates: ML={center_mlapdv[0]:.1f}μm, '
            f'AP={center_mlapdv[1]:.1f}μm, DV={center_mlapdv[2]:.1f}μm from bregma.'
        ),
        
        imaging_rate=metadata['scanImageParams']['hRoiManager']['scanFrameRate'],
        
        reference_frame=(
            'Stereotactic coordinates relative to bregma. '
            'ML (medial-lateral): positive is lateral (away from midline). '
            'AP (anterior-posterior): positive is anterior, negative is posterior. '
            'DV (dorsal-ventral): negative is ventral (deeper). '
            'Coordinates from Allen Common Coordinate Framework (CCF) 2017.'
        ),
        
        origin_coords=origin_coords,
        
        origin_coords_unit='meters',
        
        grid_spacing=grid_spacing,
        
        grid_spacing_unit='meters'
    )
    
    nwbfile.add_imaging_plane(imaging_plane)
    
    print(f"\nCreated ImagingPlane for FOV {fov_idx}:")
    print(f"  Origin: ML={origin_coords[0]*1e6:.1f}μm, AP={origin_coords[1]*1e6:.1f}μm, DV={origin_coords[2]*1e6:.1f}μm")
    print(f"  Grid spacing: {grid_spacing[0]*1e6:.3f} x {grid_spacing[1]*1e6:.3f} μm/pixel")
    print(f"  Imaging rate: {imaging_plane.imaging_rate:.2f} Hz")
```

### Field-by-Field Explanation

#### **name** (str)
```python
name = f'ImagingPlaneFOV{fov_idx:02d}'
```
- Use a descriptive name that includes the FOV index
- Examples: `'ImagingPlaneFOV00'`, `'ImagingPlaneFOV01'`, etc.

#### **optical_channel** (list or OpticalChannel)
```python
optical_channel = OpticalChannel(
    name='OpticalChannel',
    description='Green fluorescence channel (primary)',
    emission_lambda=510.0  # GCaMP emission peak
)
```
- The channel saved is channel 2 (primary green): `metadata['channelSaved']`
- Emission wavelength for GCaMP is typically ~510 nm
- Could add red channel if dual-color imaging is used

#### **description** (str)
```python
description = (
    f'Field of view {fov_idx} (UUID: {fov["roiUUID"]}). '
    f'Center location: ML={center_mlapdv[0]:.1f}μm, '
    f'AP={center_mlapdv[1]:.1f}μm, DV={center_mlapdv[2]:.1f}μm. '
    f'Allen CCF 2017 brain region ID: {brain_region_id}. '
    f'Image dimensions: {dimensions[0]}x{dimensions[1]} pixels.'
)
```
- Include FOV UUID from `fov['roiUUID']`
- Add center coordinates from `fov['MLAPDV']['center']`
- Include brain region ID from `fov['brainLocationIds']['center']`
- Mention image dimensions from `fov['nXnYnZ']`

#### **device** (Device)
```python
device = Device(
    name='Mesoscope',
    description='Two-photon mesoscope with resonant scanning',
    manufacturer='Custom IBL mesoscope'
)
```
- Create once and reference for all imaging planes
- Could include scanner frequency: 12,018.5 Hz from `metadata['scanImageParams']['hScan2D']['scannerFrequency']`

#### **excitation_lambda** (float)
```python
excitation_lambda = 920.0  # in nanometers
```
- Two-photon excitation wavelength
- Typical values for GCaMP: 920 nm or 940 nm
- Not directly in metadata, but standard for this indicator

#### **indicator** (str)
```python
indicator = 'GCaMP6s'  # or 'GCaMP6f', 'GCaMP7', etc.
```
- Calcium indicator used
- Common options: GCaMP6s (slow), GCaMP6f (fast), GCaMP7
- Should be obtained from experimental protocol

#### **location** (str)
```python
location = (
    f'Brain region ID {brain_region_id} (Allen CCF 2017). '
    f'Stereotactic coordinates: ML={center_mlapdv[0]:.1f}μm, '
    f'AP={center_mlapdv[1]:.1f}μm, DV={center_mlapdv[2]:.1f}μm from bregma.'
)
```
- Use brain region ID from `fov['brainLocationIds']['center']`
- Include stereotactic coordinates from `fov['MLAPDV']['center']`
- Common region IDs: 450 (VISp - Primary visual cortex), 981 (Retrosplenial), 1030 (VISa - Anterior visual)

#### **imaging_rate** (float)
```python
imaging_rate = metadata['scanImageParams']['hRoiManager']['scanFrameRate']
# 5.07538 Hz
```
- Frame rate in Hz from `scanFrameRate`
- This is the rate for the entire multi-ROI acquisition

#### **reference_frame** (str)
```python
reference_frame = (
    'Stereotactic coordinates relative to bregma. '
    'ML (medial-lateral): positive is lateral (away from midline). '
    'AP (anterior-posterior): positive is anterior, negative is posterior. '
    'DV (dorsal-ventral): negative is ventral (deeper). '
    'Coordinates from Allen Common Coordinate Framework (CCF) 2017.'
)
```
- Define coordinate system clearly
- IBL uses bregma as reference point
- ML: medial-lateral (left-right)
- AP: anterior-posterior (front-back)
- DV: dorsal-ventral (top-bottom)

#### **origin_coords** (array)
```python
# Top-left corner of FOV in stereotactic coordinates
top_left = fov['MLAPDV']['topLeft']  # [ML, AP, DV] in μm
origin_coords = [
    top_left[0] * 1e-6,  # Convert to meters
    top_left[1] * 1e-6,
    top_left[2] * 1e-6
]
```
- Use top-left corner from `fov['MLAPDV']['topLeft']`
- Convert from micrometers to meters
- Represents the physical location of pixel (0, 0)

#### **origin_coords_unit** (str)
```python
origin_coords_unit = 'meters'
```
- NWB default is meters
- Must convert from micrometers (as stored in metadata)

#### **grid_spacing** (array)
```python
# Calculate from corner coordinates
top_left = np.array(fov['MLAPDV']['topLeft'])
top_right = np.array(fov['MLAPDV']['topRight'])
bottom_left = np.array(fov['MLAPDV']['bottomLeft'])

width_um = np.linalg.norm(top_right - top_left)
height_um = np.linalg.norm(bottom_left - top_left)

pixel_size_x = width_um / fov['nXnYnZ'][0]
pixel_size_y = height_um / fov['nXnYnZ'][1]

grid_spacing = [
    pixel_size_x * 1e-6,  # Convert μm to meters
    pixel_size_y * 1e-6
]
```
- Calculate pixel size from FOV physical dimensions
- Width/height in μm divided by number of pixels
- Typical value: ~1.3-1.4 μm/pixel for this setup
- Convert to meters for NWB

#### **grid_spacing_unit** (str)
```python
grid_spacing_unit = 'meters'
```
- NWB default is meters
- Must match the units used in grid_spacing array

### Additional Metadata to Store

Consider adding custom fields or using `ImagingPlane` description to store:

```python
# Additional metadata that could go in description or custom fields
additional_info = {
    'fov_uuid': fov['roiUUID'],
    'z_position_um': fov['Zs'],
    'scanner_frequency_hz': metadata['scanImageParams']['hScan2D']['scannerFrequency'],
    'line_period_sec': metadata['scanImageParams']['hRoiManager']['linePeriod'],
    'flyto_time_sec': metadata['scanImageParams']['hScan2D']['flytoTimePerScanfield'],
    'field_curvature_correction': metadata['scanImageParams']['hFastZ']['enableFieldCurveCorr'],
    'brain_region_corners': {
        'top_left': fov['brainLocationIds']['topLeft'],
        'top_right': fov['brainLocationIds']['topRight'],
        'bottom_left': fov['brainLocationIds']['bottomLeft'],
        'bottom_right': fov['brainLocationIds']['bottomRight'],
        'center': fov['brainLocationIds']['center']
    },
    'line_indices': {
        'start': fov['lineIdx'][0],
        'end': fov['lineIdx'][-1],
        'total': len(fov['lineIdx'])
    }
}
```

### Validation Example

```python
# Verify the imaging plane was created correctly
for fov_idx in range(len(metadata['FOV'])):
    plane_name = f'ImagingPlaneFOV{fov_idx:02d}'
    plane = nwbfile.imaging_planes[plane_name]
    
    print(f"\n{plane_name}:")
    print(f"  Device: {plane.device.name}")
    print(f"  Excitation λ: {plane.excitation_lambda} nm")
    print(f"  Indicator: {plane.indicator}")
    print(f"  Imaging rate: {plane.imaging_rate:.2f} Hz")
    print(f"  Origin (μm): [{plane.origin_coords[0]*1e6:.1f}, "
          f"{plane.origin_coords[1]*1e6:.1f}, {plane.origin_coords[2]*1e6:.1f}]")
    print(f"  Pixel size (μm): [{plane.grid_spacing[0]*1e6:.3f}, "
          f"{plane.grid_spacing[1]*1e6:.3f}]")
    print(f"  Optical channels: {[ch.name for ch in plane.optical_channel]}")
```

### Key Considerations

1. **Multiple FOVs**: Create one `ImagingPlane` per FOV to preserve spatial organization
2. **Unit conversion**: Always convert micrometers to meters for NWB
3. **Coordinate system**: Clearly document that coordinates are stereotactic relative to bregma
4. **Brain regions**: Store Allen CCF IDs for anatomical reference
5. **Pixel spacing**: Calculate from physical FOV size and pixel dimensions
6. **Frame rate**: Use the multi-ROI frame rate, not the line rate
7. **Origin**: Use top-left corner to match image indexing convention
8. **Metadata preservation**: Store UUIDs and additional parameters in descriptions for traceability

