from pathlib import Path
from neuroconv.utils import load_dict_from_file
from typing import List
import json
import numpy as np


def update_mesoscope_ophys_metadata(
    ophys_metadata_path: Path, raw_imaging_metadata_path: Path, FOV_names: List[str]
) -> dict:
    """
    Update the ophys metadata structure with actual values from raw imaging metadata.

    This function loads a template metadata structure from a YAML file and populates it with
    actual experimental values from the IBL mesoscope raw imaging metadata JSON file. It creates
    separate metadata entries for each field of view (FOV), including imaging planes, plane
    segmentations, two-photon series, fluorescence traces, and segmentation images.

    The function performs the following steps:
    1. Load the metadata structure template from a YAML file
    2. Load actual values from `_ibl_rawImagingData.meta.json` containing:
       - ScanImage configuration parameters
       - ROI definitions and spatial coordinates
       - Laser settings and scanner properties
       - Coordinate transformations (stereotactic coordinates)
    3. For each FOV, extract and calculate:
       - Spatial parameters (origin coordinates, grid spacing, pixel size)
       - Brain region information (Allen CCF IDs)
       - Imaging parameters (frame rate, z-position, dimensions)
    4. Create metadata entries for each FOV with unique names and proper cross-references

    Parameters
    ----------
    ophys_metadata_path : Path
        Path to the YAML file containing the ophys metadata template structure.
        Expected to contain templates for: ImagingPlane, ImageSegmentation,
        TwoPhotonSeries, Fluorescence, and SegmentationImages.
    raw_imaging_metadata_path : Path
        Path to the `_ibl_rawImagingData.meta.json` file containing comprehensive
        metadata about the mesoscopic imaging acquisition. This file includes:
        - FOV array with spatial coordinates for each field of view
        - ScanImage parameters (frame rate, scanner frequency, etc.)
        - Stereotactic coordinates (ML, AP, DV) relative to bregma
        - Brain region IDs from Allen Common Coordinate Framework (CCF) 2017
    FOV_names : List[str]
        List of field of view (FOV) names to process, used for creating unique
        identifiers. Length must match the number of FOVs in raw_imaging_metadata.
        Example: ['FOV_00', 'FOV_01', 'FOV_02', ...]

    Returns
    -------
    dict
        Updated metadata dictionary with the following structure:
        {
            'Ophys': {
                'Device': [...],  # Original device info preserved
                'ImagingPlane': [  # One entry per FOV
                    {
                        'name': 'imaging_plane_FOV_00',
                        'description': '...',
                        'imaging_rate': 5.07,
                        'location': '...',
                        'origin_coords': [x, y, z],  # in meters
                        'grid_spacing': [dx, dy],     # in meters
                        ...
                    },
                    ...
                ],
                'ImageSegmentation': {
                    'plane_segmentations': [  # One entry per FOV
                        {
                            'name': 'plane_segmentation_FOV_00',
                            'imaging_plane': 'imaging_plane_FOV_00',
                            ...
                        },
                        ...
                    ]
                },
                'TwoPhotonSeries': [...],  # One entry per FOV
                'Fluorescence': {  # Dictionary keyed by plane_segmentation name
                    'plane_segmentation_FOV_00': {
                        'raw': {...},
                        'deconvolved': {...},
                        'neuropil': {...}
                    },
                    ...
                },
                'SegmentationImages': {  # Dictionary keyed by plane_segmentation name
                    'plane_segmentation_FOV_00': {
                        'mean': {...}
                    },
                    ...
                }
            }
        }

    Notes
    -----
    - Spatial coordinates are converted from micrometers (as stored in raw metadata)
      to meters (NWB standard)
    - Origin coordinates represent the top-left corner of each FOV in stereotactic space
    - Grid spacing (pixel size) is calculated from FOV physical dimensions and pixel count
    - The function uses deep copying of template structures to ensure independence
      between FOVs

    Examples
    --------
    >>> from pathlib import Path
    >>> ophys_path = Path("metadata/mesoscope_ophys_metadata.yaml")
    >>> raw_path = Path("raw_imaging_data_00/_ibl_rawImagingData.meta.json")
    >>> fov_names = ['FOV_00', 'FOV_01', 'FOV_02']
    >>> metadata = update_mesoscope_ophys_metadata(ophys_path, raw_path, fov_names)
    >>> len(metadata['Ophys']['ImagingPlane'])
    3
    >>> metadata['Ophys']['ImagingPlane'][0]['name']
    'imaging_plane_FOV_00'
    """

    # Load ophys metadata structure
    ophys_metadata = load_dict_from_file(ophys_metadata_path)

    # Load raw imaging metadata
    with open(raw_imaging_metadata_path, "r") as f:
        raw_metadata = json.load(f)

    # Get the template structures (single entries from YAML)
    imaging_plane_template = ophys_metadata["Ophys"]["ImagingPlane"][0]
    plane_seg_template = ophys_metadata["Ophys"]["ImageSegmentation"]["plane_segmentations"][0]
    two_photon_series_template = ophys_metadata["Ophys"]["TwoPhotonSeries"][0]
    fluorescence_template = ophys_metadata["Ophys"]["Fluorescence"]
    seg_images_template = ophys_metadata["Ophys"]["SegmentationImages"]

    # Clear the lists to populate with actual FOV data
    ophys_metadata["Ophys"]["ImagingPlane"] = []
    ophys_metadata["Ophys"]["ImageSegmentation"]["plane_segmentations"] = []
    ophys_metadata["Ophys"]["TwoPhotonSeries"] = []

    # Clear the Fluorescence and SegmentationImages structures
    ophys_metadata["Ophys"]["Fluorescence"] = {}
    ophys_metadata["Ophys"]["SegmentationImages"] = {}

    # Get global imaging rate
    imaging_rate = raw_metadata["scanImageParams"]["hRoiManager"]["scanFrameRate"]

    # Get device information (assumed single device)
    device_metadata = ophys_metadata["Ophys"]["Device"][0]

    # Iterate through each FOV
    for fov_idx, fov_name in enumerate(FOV_names):
        fov = raw_metadata["FOV"][fov_idx]

        # Extract FOV-specific metadata
        fov_uuid = fov["roiUUID"]
        center_mlapdv = fov["MLAPDV"]["center"]  # [ML, AP, DV] in micrometers
        brain_region_id = fov["brainLocationIds"]["center"]
        dimensions = fov["nXnYnZ"]  # [width, height, depth] in pixels

        # Calculate origin_coords from top-left corner (convert from micrometers to meters)
        top_left = fov["MLAPDV"]["topLeft"]
        origin_coords = [
            top_left[0] * 1e-6,  # ML in meters
            top_left[1] * 1e-6,  # AP in meters
            top_left[2] * 1e-6,  # DV in meters
        ]

        # Calculate grid_spacing (pixel size in meters)
        top_right = np.array(fov["MLAPDV"]["topRight"])
        bottom_left = np.array(fov["MLAPDV"]["bottomLeft"])
        top_left_array = np.array(top_left)

        width_um = np.linalg.norm(top_right - top_left_array)
        height_um = np.linalg.norm(bottom_left - top_left_array)

        pixel_size_x = width_um / dimensions[0]  # μm/pixel
        pixel_size_y = height_um / dimensions[1]  # μm/pixel

        grid_spacing = [pixel_size_x * 1e-6, pixel_size_y * 1e-6]  # x spacing in meters  # y spacing in meters

        # Create ImagingPlane entry for this FOV
        imaging_plane = imaging_plane_template.copy()
        imaging_plane["name"] = f"imaging_plane_{fov_name}"
        imaging_plane["description"] = (
            f"Field of view {fov_idx} (UUID: {fov_uuid}). "
            f"Center location: ML={center_mlapdv[0]:.1f}um, "
            f"AP={center_mlapdv[1]:.1f}um, DV={center_mlapdv[2]:.1f}um. "
            f"Allen CCF 2017 brain region ID: {brain_region_id}. "
            f"Image dimensions: {dimensions[0]}x{dimensions[1]} pixels."
        )
        imaging_plane["imaging_rate"] = imaging_rate
        imaging_plane["location"] = f"Brain region ID {brain_region_id} (Allen CCF 2017)"
        imaging_plane["origin_coords"] = origin_coords
        imaging_plane["grid_spacing"] = grid_spacing
        imaging_plane["device"] = device_metadata["name"]

        ophys_metadata["Ophys"]["ImagingPlane"].append(imaging_plane)

        # Create PlaneSegmentation entry for this FOV
        plane_seg = plane_seg_template.copy()
        plane_seg["name"] = f"plane_segmentation_{fov_name}"
        plane_seg["imaging_plane"] = f"imaging_plane_{fov_name}"

        ophys_metadata["Ophys"]["ImageSegmentation"]["plane_segmentations"].append(plane_seg)

        # Create Motion Corrected TwoPhotonSeries entry for this FOV
        mc_two_photon_series = two_photon_series_template.copy()
        mc_two_photon_series["name"] = f"motion_corrected_two_photon_series_{fov_name}"
        mc_two_photon_series["imaging_plane"] = f"imaging_plane_{fov_name}"

        ophys_metadata["Ophys"]["TwoPhotonSeries"].append(mc_two_photon_series)

        # Create Fluorescence entries for this FOV
        plane_seg_key = f"plane_segmentation_{fov_name}"
        ophys_metadata["Ophys"]["Fluorescence"][plane_seg_key] = {
            "raw": {
                "name": f"raw_response_series_{fov_name}",
                "description": fluorescence_template["plane_segmentation"]["raw"]["description"],
                "unit": fluorescence_template["plane_segmentation"]["raw"]["unit"],
            },
            "deconvolved": {
                "name": f"deconvolved_response_series_{fov_name}",
                "description": fluorescence_template["plane_segmentation"]["deconvolved"]["description"],
                "unit": fluorescence_template["plane_segmentation"]["deconvolved"]["unit"],
            },
            "neuropil": {
                "name": f"neuropil_response_series_{fov_name}",
                "description": fluorescence_template["plane_segmentation"]["neuropil"]["description"],
                "unit": fluorescence_template["plane_segmentation"]["neuropil"]["unit"],
            },
        }

        # Create SegmentationImages entries for this FOV
        ophys_metadata["Ophys"]["SegmentationImages"][plane_seg_key] = {
            "mean": {
                "name": f"mean_image_{fov_name}",
                "description": seg_images_template["plane_segmentation"]["mean"]["description"],
            }
        }

    return ophys_metadata


if __name__ == "__main__":
    ophys_metadata_path = Path(
        r"C:\Users\amtra\CatalystNeuro\IBL-mesoscope-to-nwb\src\ibl_mesoscope_to_nwb\mesoscope2025\metadata\mesoscope_ophys_metadata.yaml"
    )
    raw_imaging_metadata_path = Path(
        r"E:\IBL-data-share\cortexlab\Subjects\SP061\2025-01-28\001\raw_imaging_data_00\_ibl_rawImagingData.meta.json"
    )
    FOV_names = ["FOV_00", "FOV_01", "FOV_02", "FOV_03", "FOV_04", "FOV_05", "FOV_06", "FOV_07"]
    updated_metadata = update_mesoscope_ophys_metadata(ophys_metadata_path, raw_imaging_metadata_path, FOV_names)
    print(updated_metadata)
