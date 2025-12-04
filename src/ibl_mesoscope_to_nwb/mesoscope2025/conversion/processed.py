"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import datetime
import json
from pathlib import Path
from typing import List, Union
from zoneinfo import ZoneInfo

import numpy as np
from neuroconv.utils import dict_deep_update, load_dict_from_file

from ibl_mesoscope_to_nwb.mesoscope2025 import ProcessedMesoscopeNWBConverter
from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import (
    IBLMesoscopeMotionCorrectedImagingExtractor,
    IBLMesoscopeSegmentationExtractor,
)


def update_processed_ophys_metadata(
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
                'ImageSegmentation': {
                    'plane_segmentations': [  # One entry per FOV
                        {
                            'name': 'PlaneSegmentationFOV00',
                            'imaging_plane': 'ImagingPlaneFOV00',
                            ...
                        },
                        ...
                    ]
                },
                'TwoPhotonSeries': [...],  # One entry per FOV
                'Fluorescence': {  # Dictionary keyed by plane_segmentation name
                    'PlaneSegmentationFOV00': {
                        'raw': {...},
                        'deconvolved': {...},
                        'neuropil': {...}
                    },
                    ...
                },
                'SegmentationImages': {  # Dictionary keyed by plane_segmentation name
                    'PlaneSegmentationFOV00': {
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
    >>> FOV_names = ['FOV_00', 'FOV_01', 'FOV_02']
    >>> metadata = update_processed_ophys_metadata(ophys_path, raw_path, FOV_names)
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
        imaging_plane["location"] = f"FOV center location: Brain region ID {brain_region_id} (Allen CCF 2017)"
        imaging_plane["origin_coords"] = origin_coords
        imaging_plane["grid_spacing"] = grid_spacing
        imaging_plane["device"] = device_metadata["name"]

        ophys_metadata["Ophys"]["ImagingPlane"].append(imaging_plane)

        # Create PlaneSegmentation entry for this FOV
        plane_seg = plane_seg_template.copy()
        plane_seg["name"] = f"PlaneSegmentation{camel_case_FOV_name}"
        plane_seg["description"] = f"Spatial components of segmented ROIs for {FOV_name} (UUID: {fov_uuid})."
        plane_seg["imaging_plane"] = f"ImagingPlane{camel_case_FOV_name}"

        ophys_metadata["Ophys"]["ImageSegmentation"]["plane_segmentations"].append(plane_seg)

        # Create Motion Corrected TwoPhotonSeries entry for this FOV
        mc_two_photon_series = two_photon_series_template.copy()
        mc_two_photon_series["name"] = f"MotionCorrectedTwoPhotonSeries{camel_case_FOV_name}"
        mc_two_photon_series["description"] = (
            f"The motion corrected two-photon imaging data acquired using the mesoscope on {FOV_name} (UUID: {fov_uuid})."
        )
        mc_two_photon_series["imaging_plane"] = f"ImagingPlane{camel_case_FOV_name}"

        ophys_metadata["Ophys"]["TwoPhotonSeries"].append(mc_two_photon_series)

        # Create Fluorescence entries for this FOV
        plane_seg_key = f"PlaneSegmentation{camel_case_FOV_name}"
        ophys_metadata["Ophys"]["Fluorescence"][plane_seg_key] = {
            "raw": {
                "name": f"RawROIResponseSeries{camel_case_FOV_name}",
                "description": f"The raw GCaMP fluorescence traces (temporal components) of segmented ROIs for {FOV_name} (UUID: {fov_uuid}).",
                "unit": fluorescence_template["plane_segmentation"]["raw"]["unit"],
            },
            "deconvolved": {
                "name": f"DeconvolvedROIResponseSeries{camel_case_FOV_name}",
                "description": f"The deconvolved activity traces (temporal components) of segmented ROIs for {FOV_name} (UUID: {fov_uuid}).",
                "unit": fluorescence_template["plane_segmentation"]["deconvolved"]["unit"],
            },
            "neuropil": {
                "name": f"NeuropilResponseSeries{camel_case_FOV_name}",
                "description": f"The neuropil signals (temporal components) for {FOV_name} (UUID: {fov_uuid}).",
                "unit": fluorescence_template["plane_segmentation"]["neuropil"]["unit"],
            },
        }

        # Create SegmentationImages entries for this FOV
        ophys_metadata["Ophys"]["SegmentationImages"][plane_seg_key] = {
            "mean": {
                "name": f"MeanImage{camel_case_FOV_name}",
                "description": f"The mean image for {FOV_name} (UUID: {fov_uuid}).",
            }
        }

    return ophys_metadata


def processed_session_to_nwb(
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    subject_id: str,
    eid: str,
    stub_test: bool = False,
    overwrite: bool = False,
):
    """
    Convert a processed IBL mesoscope session to NWB format.

    This function converts processed mesoscope data including motion-corrected
    imaging and segmentation data from the IBL pipeline to NWB format.

    Parameters
    ----------
    data_dir_path : Union[str, Path]
        Path to the directory containing the processed session data.
        Expected to contain 'suite2p' and 'alf' subdirectories.
    output_dir_path : Union[str, Path]
        Path to the directory where the NWB file will be saved.
    subject_id : str
        The subject ID for the session (e.g., 'SP061').
    eid : str
        The experiment ID (session ID) for the session.
    stub_test : bool, optional
        Whether to run a stub test with limited data (first 2 planes only),
        by default False.
    overwrite : bool, optional
        Whether to overwrite existing NWB files, by default False.
    """
    data_dir_path = Path(data_dir_path)
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    nwbfile_path = output_dir_path / f"sub-{subject_id}_ses-{eid}.nwb"

    source_data = dict()
    conversion_options = dict()

    # Add Motion Corrected Imaging
    mc_imaging_folder = data_dir_path / "suite2p"
    if not mc_imaging_folder.exists():
        mc_imaging_folder = data_dir_path / "suite2"  # correct for typo in folder name
        if not mc_imaging_folder.exists():
            raise FileNotFoundError(f"Motion corrected imaging folder not found at {mc_imaging_folder}")
    available_FOVs = IBLMesoscopeMotionCorrectedImagingExtractor.get_available_planes(mc_imaging_folder)
    available_FOVs = available_FOVs[:2] if stub_test else available_FOVs  # Limit to first 2 planes for testing
    for FOV_index, FOV_name in enumerate(available_FOVs):
        file_path = mc_imaging_folder / FOV_name / "imaging.frames_motionRegistered.bin"
        source_data.update({f"{FOV_name}MotionCorrectedImaging": dict(file_path=file_path)})
        conversion_options.update(
            {f"{FOV_name}MotionCorrectedImaging": dict(stub_test=False, photon_series_index=FOV_index)}
        )

    # Add Segmentation
    alf_folder = data_dir_path / "alf"
    FOV_names = IBLMesoscopeSegmentationExtractor.get_available_planes(alf_folder)
    FOV_names = FOV_names[:2] if stub_test else FOV_names  # Limit to first 2 planes for testing
    for FOV_name in FOV_names:
        source_data.update({f"{FOV_name}Segmentation": dict(folder_path=alf_folder, FOV_name=FOV_name)})
        conversion_options.update({f"{FOV_name}Segmentation": dict(stub_test=False)})

    # Add Anatomical Localization
    for FOV_name in FOV_names:
        source_data.update({f"{FOV_name}AnatomicalLocalization": dict(folder_path=alf_folder, FOV_name=FOV_name)})
        conversion_options.update({f"{FOV_name}AnatomicalLocalization": dict()})

    # Add Lick Times
    source_data.update({"LickTimes": dict(folder_path=alf_folder)})
    conversion_options.update({"LickTimes": dict()})

    # Add Wheel Movement
    source_data.update({"WheelMovement": dict(folder_path=alf_folder / "task_00")})
    conversion_options.update({"WheelMovement": dict(stub_test=stub_test)})

    # Add ROI Motion Energy
    camera_names = ["rightCamera", "leftCamera"]
    for camera_name in camera_names:
        source_data.update({f"{camera_name}ROIMotionEnergy": dict(folder_path=alf_folder, camera_name=camera_name)})
        conversion_options.update({f"{camera_name}ROIMotionEnergy": dict()})

    # Add Pupil Tracking
    for camera_name in camera_names:
        source_data.update({f"{camera_name}PupilTracking": dict(folder_path=alf_folder, camera_name=camera_name)})
        conversion_options.update({f"{camera_name}PupilTracking": dict()})

    converter = ProcessedMesoscopeNWBConverter(source_data=source_data)

    # Add datetime to conversion
    metadata = converter.get_metadata()
    date = datetime.datetime(year=2020, month=1, day=1, tzinfo=ZoneInfo("US/Eastern"))
    metadata["NWBFile"]["session_start_time"] = date

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent.parent / "metadata" / "mesoscope_general_metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    # Update ophys metadata
    ophys_metadata_path = Path(__file__).parent.parent / "metadata" / "mesoscope_processed_ophys_metadata.yaml"
    raw_imaging_metadata_path = data_dir_path / "raw_imaging_data_00" / "_ibl_rawImagingData.meta.json"
    updated_ophys_metadata = update_processed_ophys_metadata(
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
