"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import json
import time
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import numpy as np
from neuroconv.utils import dict_deep_update, load_dict_from_file
from one.api import ONE

from ibl_mesoscope_to_nwb.mesoscope2025 import ProcessedMesoscopeNWBConverter
from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import (
    IBLMesoscopeAnatomicalLocalizationInterface,
    IBLMesoscopeMotionCorrectedImagingExtractor,
    IBLMesoscopeMotionCorrectedImagingInterface,
    IBLMesoscopeSegmentationExtractor,
    IBLMesoscopeSegmentationInterface,
)


def _get_processed_data_interfaces(
    one: ONE,
    eid: str,
) -> dict:
    """
    Returns a dictionary of data interfaces for processed behavior data for a given session.

    Parameters
    ----------
    one: ONE
        An instance of the ONE API to access data.
    eid: str
        The session ID.

    Returns
    -------
    dict
        A dictionary where keys are interface names and values are corresponding data interface instances.
    """

    try:
        from ibl_to_nwb.bwm_to_nwb import get_camera_name_from_file
        from ibl_to_nwb.datainterfaces import (
            BrainwideMapTrialsInterface,
            LickInterface,
            PupilTrackingInterface,
            RoiMotionEnergyInterface,
            SessionEpochsInterface,
            WheelInterface,
        )
    except ImportError as e:
        raise ImportError(
            "ibl_to_nwb is required for processed behavior conversion. "
            # TODO update URL
            "Please install it from https://github.com/h-mayorquin/IBL-to-nwb/blob/heberto_conversion."
        ) from e

    # TODO Add data interfaces specific for mesoscope dataset
    # Motion Corrected Imaging
    # Segmentation
    # Anatomical Localization

    data_interfaces = dict()
    interface_kwargs = dict(one=one, session=eid)

    data_interfaces["BrainwideMapTrials"] = BrainwideMapTrialsInterface(**interface_kwargs)
    data_interfaces["Wheel"] = WheelInterface(**interface_kwargs)

    if one.list_datasets(eid=eid, collection="alf", filename="licks*"):
        data_interfaces["Lick"] = LickInterface(**interface_kwargs)

    pupil_tracking_files = one.list_datasets(eid=eid, filename="*features*")
    for pupil_tracking_file in pupil_tracking_files:
        camera_name = get_camera_name_from_file(pupil_tracking_file)
        if PupilTrackingInterface.check_availability(one=one, eid=eid, camera_name=camera_name)["available"]:
            data_interfaces[f"PupilTracking_{camera_name}"] = PupilTrackingInterface(
                camera_name=camera_name, **interface_kwargs
            )
        else:
            print(f"Pupil tracking data for camera '{camera_name}' not available or failed to load, skipping...")

    roi_motion_energy_files = one.list_datasets(eid=eid, filename="*ROIMotionEnergy.npy*")
    for roi_motion_energy_file in roi_motion_energy_files:
        camera_name = get_camera_name_from_file(roi_motion_energy_file)
        if RoiMotionEnergyInterface.check_availability(one=one, eid=eid, camera_name=camera_name)["available"]:
            data_interfaces[f"RoiMotionEnergy_{camera_name}"] = RoiMotionEnergyInterface(
                camera_name=camera_name, **interface_kwargs
            )
        else:
            print(f"ROI motion energy data for camera '{camera_name}' not available or failed to load, skipping...")

    # Session epochs (high-level task vs passive phases)
    if SessionEpochsInterface.check_availability(one, eid)["available"]:
        data_interfaces["SessionEpochs"] = SessionEpochsInterface(one=one, session=eid)

    return data_interfaces


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
    output_dir_path: str | Path,
    data_dir_path: str | Path,
    one_api_kwargs: dict,
    stub_test: bool = False,
    append_on_disk_nwbfile: bool = False,
):
    """
    Convert a processed IBL mesoscope session to NWB format.

    This function converts processed mesoscope data including motion-corrected
    imaging and segmentation data from the IBL pipeline to NWB format.

    Expected file structure:
    data_dir_path/
      ├──  suite2p/
      │     ├──  plane0/
      │     │     └──  imaging.frames_motionRegistered.bin
      ├──  .npy
      ├──  .htsv
      ├──  .npy
      ├──  .npy
      ├──  .npy
      └──  .npy

    Parameters
    ----------
    output_dir_path: str | Path
        Path to the directory where the output NWB file will be saved.
    data_dir_path: str | Path
        Path to the directory containing the processed widefield imaging data.
    one_api_kwargs: dict
        Keyword arguments to initialize the interfaces that require ONE API access.
    stub_test: bool, default: False
        Whether to run a stub test (process a smaller subset of data for testing purposes).
    append_on_disk_nwbfile: bool, default: False
        If True, append data to an existing on-disk NWB file instead of creating a new one.
    """
    data_dir_path = Path(data_dir_path)
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    data_interfaces = dict()
    conversion_options = dict()

    # # Add Motion Corrected Imaging
    mc_imaging_folder = data_dir_path / "suite2p"
    if not mc_imaging_folder.exists():
        mc_imaging_folder = data_dir_path / "suite2"  # correct for typo in folder name
        if not mc_imaging_folder.exists():
            raise FileNotFoundError(f"Motion corrected imaging folder not found at {mc_imaging_folder}")
    available_FOVs = IBLMesoscopeMotionCorrectedImagingExtractor.get_available_planes(mc_imaging_folder)
    available_FOVs = available_FOVs[:2] if stub_test else available_FOVs  # Limit to first 2 planes for testing
    for FOV_index, FOV_name in enumerate(available_FOVs):
        file_path = mc_imaging_folder / FOV_name / "imaging.frames_motionRegistered.bin"
        data_interfaces[f"{FOV_name}MotionCorrectedImaging"] = IBLMesoscopeMotionCorrectedImagingInterface(
            file_path=file_path
        )
        conversion_options.update(
            {f"{FOV_name}MotionCorrectedImaging": dict(stub_test=False, photon_series_index=FOV_index)}
        )

    # # Add Segmentation
    alf_folder = data_dir_path / "alf"
    FOV_names = IBLMesoscopeSegmentationExtractor.get_available_planes(alf_folder)
    FOV_names = FOV_names[:2] if stub_test else FOV_names  # Limit to first 2 planes for testing
    for FOV_name in FOV_names:
        data_interfaces[f"{FOV_name}Segmentation"] = IBLMesoscopeSegmentationInterface(
            folder_path=alf_folder, FOV_name=FOV_name
        )
        conversion_options.update({f"{FOV_name}Segmentation": dict(stub_test=False)})

    # Add Anatomical Localization
    for FOV_name in FOV_names:
        data_interfaces[f"{FOV_name}AnatomicalLocalization"] = IBLMesoscopeAnatomicalLocalizationInterface(
            folder_path=alf_folder, FOV_name=FOV_name
        )
        conversion_options.update({f"{FOV_name}AnatomicalLocalization": dict()})

    # Add Behavior
    behavior_interfaces = _get_processed_data_interfaces(**one_api_kwargs)
    data_interfaces.update(behavior_interfaces)

    converter = ProcessedMesoscopeNWBConverter(**one_api_kwargs, data_interfaces=data_interfaces)

    # Add datetime to conversion
    metadata = converter.get_metadata()
    session_start_time = metadata["NWBFile"]["session_start_time"]
    if session_start_time.tzinfo is None:
        session_start_time = session_start_time.replace(tzinfo=ZoneInfo("US/Eastern"))
    metadata["NWBFile"]["session_start_time"] = session_start_time

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent.parent / "_metadata" / "mesoscope_general_metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    # Update ophys metadata
    ophys_metadata_path = Path(__file__).parent.parent / "_metadata" / "mesoscope_processed_ophys_metadata.yaml"
    raw_imaging_metadata_path = data_dir_path / "raw_imaging_data_00" / "_ibl_rawImagingData.meta.json"
    updated_ophys_metadata = update_processed_ophys_metadata(
        ophys_metadata_path=ophys_metadata_path,
        raw_imaging_metadata_path=raw_imaging_metadata_path,
        FOV_names=FOV_names,
    )
    metadata = dict_deep_update(metadata, updated_ophys_metadata)

    subject_id = metadata["Subject"]["subject_id"]
    fname = f"sub-{subject_id}_ses-{one_api_kwargs['eid']}_desc-processed_behavior+ophys.nwb"
    nwbfile_path = Path(output_dir_path) / fname

    overwrite = False
    if nwbfile_path.exists() and not append_on_disk_nwbfile:
        overwrite = True

    print(f"Writing to NWB '{nwbfile_path}' ...")
    conversion_start = time.time()

    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        append_on_disk_nwbfile=append_on_disk_nwbfile,
        overwrite=overwrite,
    )

    conversion_time = time.time() - conversion_start

    # Calculate total size
    total_size_bytes = nwbfile_path.stat().st_size
    total_size_gb = total_size_bytes / (1024**3)

    print(f"Conversion completed in {int(conversion_time // 60)}:{conversion_time % 60:05.2f} (MM:SS.ss)")
    print(f"Total data ({nwbfile_path.name}) size: {total_size_gb:.2f} GB ({total_size_bytes:,} bytes)")

    return nwbfile_path


if __name__ == "__main__":
    # Example usage
    data_dir_path = Path(r"E:\IBL-data-share\cortexlab\Subjects\SP061\2025-01-28\001")
    output_dir_path = Path(r"E:\ibl_mesoscope_conversion_nwb")
    one_api_kwargs = {
        "one": ONE(base_url="https://alyx.internationalbrainlab.org", silent=False),
        "eid": "5ce2e17e-8471-42d4-8a16-21949710b328",
    }
    processed_session_to_nwb(
        output_dir_path=output_dir_path,
        data_dir_path=data_dir_path,
        one_api_kwargs=one_api_kwargs,
        stub_test=True,
        append_on_disk_nwbfile=False,
    )
