"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import datetime
import json
from pathlib import Path
from typing import List, Union
from zoneinfo import ZoneInfo

import numpy as np
from natsort import natsorted
from neuroconv.utils import dict_deep_update, load_dict_from_file

from ibl_mesoscope_to_nwb.mesoscope2025 import RawMesoscopeNWBConverter


def update_raw_ophys_metadata(ophys_metadata_path: Path, raw_imaging_metadata_path: Path, FOV_names: List[str]) -> dict:
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
                        'name': 'ImagingPlaneFOV00',
                        'description': '...',
                        'imaging_rate': 5.07,
                        'location': '...',
                        'origin_coords': [x, y, z],  # in meters
                        'grid_spacing': [dx, dy],     # in meters
                        ...
                    },
                    ...
                ],
                'TwoPhotonSeries': [...],  # One entry per FOV
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
    >>> ophys_path = Path("metadata/mesoscope_raw_ophys_metadata.yaml")
    >>> raw_path = Path("raw_imaging_data_00/_ibl_rawImagingData.meta.json")
    >>> FOV_names = ['FOV_00', 'FOV_01', 'FOV_02']
    >>> metadata = update_raw_ophys_metadata(ophys_path, raw_path, FOV_names)
    >>> len(metadata['Ophys']['ImagingPlane'])
    3
    >>> metadata['Ophys']['ImagingPlane'][0]['name']
    'ImagingPlaneFOV00'
    """

    # Load ophys metadata structure
    ophys_metadata = load_dict_from_file(ophys_metadata_path)

    # Load raw imaging metadata
    with open(raw_imaging_metadata_path, "r") as f:
        raw_metadata = json.load(f)

    # Get the template structures (single entries from YAML)
    imaging_plane_template = ophys_metadata["Ophys"]["ImagingPlane"][0]
    two_photon_series_template = ophys_metadata["Ophys"]["TwoPhotonSeries"][0]

    # Clear the lists to populate with actual FOV data
    ophys_metadata["Ophys"]["ImagingPlane"] = []
    ophys_metadata["Ophys"]["TwoPhotonSeries"] = []

    # Get global imaging rate
    imaging_rate = raw_metadata["scanImageParams"]["hRoiManager"]["scanFrameRate"]

    # Get device information (assumed single device)
    device_metadata = ophys_metadata["Ophys"]["Device"][0]

    # Iterate through each FOV
    for FOV_index, FOV_name in enumerate(FOV_names):
        camel_case_FOV_name = FOV_name.replace("_", "")
        fov = raw_metadata["FOV"][FOV_index]

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
        imaging_plane["name"] = f"ImagingPlane{camel_case_FOV_name}"
        imaging_plane["description"] = (
            f"Field of view {FOV_index} (UUID: {fov_uuid}). "
            f"Center location: ML={center_mlapdv[0]:.1f}um, "
            f"AP={center_mlapdv[1]:.1f}um, DV={center_mlapdv[2]:.1f}um. "
            f"Image dimensions: {dimensions[0]}x{dimensions[1]} pixels."
        )
        imaging_plane["imaging_rate"] = imaging_rate
        imaging_plane["location"] = f"Brain region ID {brain_region_id} (Allen CCF 2017)"
        imaging_plane["origin_coords"] = origin_coords
        imaging_plane["grid_spacing"] = grid_spacing
        imaging_plane["device"] = device_metadata["name"]

        ophys_metadata["Ophys"]["ImagingPlane"].append(imaging_plane)

        # Create TwoPhotonSeries entry for this FOV
        two_photon_series = two_photon_series_template.copy()
        two_photon_series["name"] = f"TwoPhotonSeries{camel_case_FOV_name}"
        two_photon_series["description"] = (
            f"The raw two-photon imaging data acquired using the mesoscope on {FOV_name} (UUID: {fov_uuid}) ."
        )
        two_photon_series["imaging_plane"] = f"ImagingPlane{camel_case_FOV_name}"

        ophys_metadata["Ophys"]["TwoPhotonSeries"].append(two_photon_series)

    return ophys_metadata


def raw_session_to_nwb(
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    subject_id: str,
    eid: str,
    stub_test: bool = False,
    overwrite: bool = False,
):
    """
    Convert a raw imaging session to NWB format.

    Parameters
    ----------
    data_dir_path : Union[str, Path]
        Path to the directory containing the raw data.
    output_dir_path : Union[str, Path]
        Path to the directory where the NWB file will be saved.
    subject_id : str
        Subject identifier.
    eid : str
        Experiment identifier.
    stub_test : bool, optional
        If True, convert only a subset of data for testing, by default False.
    overwrite : bool, optional
        If True, overwrite existing NWB file, by default False.
    """
    data_dir_path = Path(data_dir_path)
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    nwbfile_path = output_dir_path / f"sub-{subject_id}_ses-{eid}.nwb"

    source_data = dict()
    conversion_options = dict()

    # Add raw imaging data
    raw_imaging_folder = data_dir_path / "raw_imaging_data_00"
    raw_imaging_metadata_path = raw_imaging_folder / "_ibl_rawImagingData.meta.json"
    with open(raw_imaging_metadata_path, "r") as f:
        raw_metadata = json.load(f)
    num_planes = len(raw_metadata["FOV"])
    FOV_names = [f"FOV_{i:02d}" for i in range(num_planes)]

    tiff_files = natsorted(raw_imaging_folder.glob(f"imaging.frames/*{subject_id}*.tif"))
    for plane_index, FOV_name in enumerate(FOV_names[:2]):  # Limiting to first 2 FOVs for testing
        source_data.update(
            {
                f"{FOV_name}RawImaging": dict(
                    file_paths=tiff_files,
                    plane_index=plane_index,
                    channel_name="Channel 1",
                    FOV_name=FOV_name,
                )
            }
        )
        conversion_options.update({f"{FOV_name}RawImaging": dict(stub_test=stub_test, photon_series_index=plane_index)})

    converter = RawMesoscopeNWBConverter(source_data=source_data)

    # Add datetime to conversion
    metadata = converter.get_metadata()
    date = datetime.datetime(year=2020, month=1, day=1, tzinfo=ZoneInfo("US/Eastern"))
    metadata["NWBFile"]["session_start_time"] = date

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent.parent / "metadata" / "mesoscope_general_metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    # # Update ophys metadata
    ophys_metadata_path = Path(__file__).parent.parent / "metadata" / "mesoscope_raw_ophys_metadata.yaml"
    updated_ophys_metadata = update_raw_ophys_metadata(
        ophys_metadata_path=ophys_metadata_path,
        raw_imaging_metadata_path=raw_imaging_metadata_path,
        FOV_names=FOV_names,
    )
    metadata = dict_deep_update(metadata, updated_ophys_metadata)

    metadata["Subject"]["subject_id"] = subject_id

    # Run conversion
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        overwrite=overwrite,
    )
