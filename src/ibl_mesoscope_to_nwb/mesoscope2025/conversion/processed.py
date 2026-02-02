"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import json
import time
from pathlib import Path
from typing import List

import numpy as np
from ibl_to_nwb.datainterfaces import (
    BrainwideMapTrialsInterface,
    IblPoseEstimationInterface,
    LickInterface,
    PassiveIntervalsInterface,
    PassiveReplayStimInterface,
    PupilTrackingInterface,
    RoiMotionEnergyInterface,
    SessionEpochsInterface,
)
from ndx_ibl import IblMetadata, IblSubject
from neuroconv.utils import dict_deep_update, load_dict_from_file
from one.api import ONE
from pynwb import NWBFile

from ibl_mesoscope_to_nwb.mesoscope2025 import ProcessedMesoscopeNWBConverter
from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import (
    IBLMesoscopeAnatomicalLocalizationInterface,
    IBLMesoscopeMotionCorrectedImagingExtractor,
    IBLMesoscopeMotionCorrectedImagingInterface,
    IBLMesoscopeSegmentationExtractor,
    IBLMesoscopeSegmentationInterface,
    MesoscopeWheelKinematicsInterface,
    MesoscopeWheelMovementsInterface,
    MesoscopeWheelPositionInterface,
)
from ibl_mesoscope_to_nwb.mesoscope2025.utils import (
    get_available_tasks,
    sanitize_subject_id_for_dandi,
    setup_paths,
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


def convert_processed_session(
    eid: str,
    one: ONE,
    base_path: Path,
    stub_test: bool = False,
    append_on_disk_nwbfile: bool = False,
    verbose: bool = True,
) -> dict:
    """Convert IBL processed session to NWB.

    Parameters
    ----------
    eid : str
        Experiment ID (session UUID)
    one : ONE
        ONE API instance
    stub_test : bool, optional
        If True, creates minimal NWB for testing without downloading large files.
        In stub mode, spike properties (spike_amplitudes, spike_distances_from_probe_tip)
        are automatically skipped to reduce memory usage.
    base_path : Path, optional
        Base output directory for NWB files
    append_on_disk_nwbfile: bool, optional
        If True, append to an existing on-disk NWB file instead of creating a new one.
    Returns
    -------
    dict
        Conversion result information including NWB file path and timing
    """
    if verbose:
        print(f"Starting PROCESSED conversion for session {eid}...")

    # Setup paths
    start_time = time.time()
    paths = setup_paths(one, eid, base_path=base_path)

    session_info = one.alyx.rest("sessions", "read", id=eid)
    subject_nickname = session_info.get("subject")
    if isinstance(subject_nickname, dict):
        subject_nickname = subject_nickname.get("nickname") or subject_nickname.get("name")
    if not subject_nickname:
        subject_nickname = "unknown"

    # New structure: nwbfiles/{full|stub}/sub-{subject}/*.nwb
    conversion_type = "stub" if stub_test else "full"
    # Sanitize subject nickname for DANDI compliance (replace underscores with hyphens)
    subject_id_for_filenames = sanitize_subject_id_for_dandi(subject_nickname)
    output_dir = Path(paths["output_folder"]) / conversion_type / f"sub-{subject_id_for_filenames}"
    output_dir.mkdir(parents=True, exist_ok=True)
    nwbfile_path = output_dir / f"sub-{subject_id_for_filenames}_ses-{eid}_desc-processed_behavior+ophys.nwb"

    # ========================================================================
    # STEP 1: Define data interfaces
    # ========================================================================

    if verbose:
        print(f"Creating data interfaces...")
    interface_creation_start = time.time()

    data_interfaces = dict()
    conversion_options = dict()
    interface_kwargs = dict(one=one, session=eid)

    # # Add Motion Corrected Imaging
    mc_imaging_folder = Path(paths["mc_imaging_folder"])
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
    alf_folder = paths["session_folder"] / "alf"
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

    # Behavioral data
    data_interfaces["BrainwideMapTrials"] = BrainwideMapTrialsInterface(**interface_kwargs)
    conversion_options.update({"BrainwideMapTrials": dict(stub_test=stub_test, stub_trials=10)})

    # Wheel data - add each interface if its data is available
    available_tasks = get_available_tasks(**interface_kwargs)
    for task in available_tasks:
        if MesoscopeWheelPositionInterface.check_availability(one, eid, task=task)["available"]:
            data_interfaces[f"{task.replace('task_', 'Task')}WheelPosition"] = MesoscopeWheelPositionInterface(
                **interface_kwargs, task=task
            )
            conversion_options.update({f"{task.replace('task_', 'Task')}WheelPosition": dict(stub_test=stub_test)})
        if MesoscopeWheelKinematicsInterface.check_availability(one, eid, task=task)["available"]:
            data_interfaces[f"{task.replace('task_', 'Task')}WheelKinematics"] = MesoscopeWheelKinematicsInterface(
                **interface_kwargs, task=task
            )
            conversion_options.update({f"{task.replace('task_', 'Task')}WheelKinematics": dict(stub_test=stub_test)})
        if MesoscopeWheelMovementsInterface.check_availability(one, eid, task=task)["available"]:
            data_interfaces[f"{task.replace('task_', 'Task')}WheelMovements"] = MesoscopeWheelMovementsInterface(
                **interface_kwargs, task=task
            )
            conversion_options.update({f"{task.replace('task_', 'Task')}WheelMovements": dict(stub_test=stub_test)})

    # Session epochs (high-level task vs passive phases)
    if SessionEpochsInterface.check_availability(one, eid)["available"]:
        data_interfaces["SessionEpochs"] = SessionEpochsInterface(**interface_kwargs)

    # Passive period data - add each interface if its data is available
    if PassiveIntervalsInterface.check_availability(one, eid)["available"]:
        data_interfaces["PassiveIntervals"] = PassiveIntervalsInterface(**interface_kwargs)

    # NOTE: PassiveRFMInterface is temporarily disabled due to data quality issues - waiting for upstream fix
    # if PassiveRFMInterface.check_availability(one, eid)["available"]:
    #     data_interfaces["PassiveRFM"] = PassiveRFMInterface(**interface_kwargs)

    if PassiveReplayStimInterface.check_availability(one, eid)["available"]:
        data_interfaces["PassiveReplayStim"] = PassiveReplayStimInterface(**interface_kwargs)

    # Licks - optional interface
    if LickInterface.check_availability(one, eid)["available"]:
        data_interfaces["Licks"] = LickInterface(**interface_kwargs)

    # Camera-based interfaces (pose estimation, pupil tracking, ROI motion energy)
    # Check availability per camera since not all sessions have all cameras
    for camera_view in ["left", "right", "body"]:
        camera_name = f"{camera_view}Camera"

        # Pose estimation - check_availability handles Lightning Pose → DLC fallback
        pose_availability = IblPoseEstimationInterface.check_availability(one, eid, camera_name=camera_name)
        if pose_availability["available"]:
            # Determine tracker from which alternative was found
            alternative = pose_availability.get("alternative_used", "lightning_pose")
            tracker = "lightningPose" if alternative == "lightning_pose" else "dlc"
            data_interfaces[f"{camera_name}PoseEstimation"] = IblPoseEstimationInterface(
                camera_name=camera_name, tracker=tracker, **interface_kwargs
            )

        # Pupil tracking - only for left/right cameras (body camera doesn't capture eyes)
        if camera_view in ["left", "right"]:
            if PupilTrackingInterface.check_availability(one, eid, camera_name=camera_name)["available"]:
                data_interfaces[f"{camera_name}PupilTracking"] = PupilTrackingInterface(
                    camera_name=camera_name, **interface_kwargs
                )

        # ROI motion energy
        if RoiMotionEnergyInterface.check_availability(one, eid, camera_name=camera_name)["available"]:
            data_interfaces[f"{camera_name}RoiMotionEnergy"] = RoiMotionEnergyInterface(
                camera_name=camera_name, **interface_kwargs
            )
    interface_creation_time = time.time() - interface_creation_start
    if verbose:
        print(f"Data interfaces created in {interface_creation_time:.2f}s")

    # ========================================================================
    # STEP 2: Create converter
    # ========================================================================
    converter = ProcessedMesoscopeNWBConverter(one=one, session=eid, data_interfaces=data_interfaces)

    # ========================================================================
    # STEP 3: Get metadata
    # ========================================================================
    metadata = converter.get_metadata()

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent.parent / "_metadata" / "mesoscope_general_metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    # Update ophys metadata
    ophys_metadata_path = Path(__file__).parent.parent / "_metadata" / "mesoscope_processed_ophys_metadata.yaml"
    raw_imaging_metadata_path = paths["session_folder"] / "raw_imaging_data_00" / "_ibl_rawImagingData.meta.json"
    updated_ophys_metadata = update_processed_ophys_metadata(
        ophys_metadata_path=ophys_metadata_path,
        raw_imaging_metadata_path=raw_imaging_metadata_path,
        FOV_names=FOV_names,
    )
    metadata = dict_deep_update(metadata, updated_ophys_metadata)

    # ========================================================================
    # STEP 4: Write NWB file
    # ========================================================================
    overwrite = False
    if nwbfile_path.exists() and not append_on_disk_nwbfile:
        overwrite = True

    subject_metadata_for_ndx = metadata.pop("Subject")
    ibl_subject = IblSubject(**subject_metadata_for_ndx)

    # TODO: Solve this for append_on_disk_nwbfile=True case
    nwbfile = NWBFile(**metadata["NWBFile"])
    nwbfile.subject = ibl_subject

    if verbose:
        print(f"Writing to NWB '{nwbfile_path}' ...")
    write_start = time.time()

    converter.run_conversion(
        metadata=metadata,
        nwbfile=nwbfile,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        append_on_disk_nwbfile=append_on_disk_nwbfile,
        overwrite=overwrite,
    )

    write_time = time.time() - write_start

    # Get NWB file size
    nwb_size_bytes = nwbfile_path.stat().st_size
    nwb_size_gb = nwb_size_bytes / (1024**3)

    if verbose:
        total_time_seconds = time.time() - start_time
        total_time_hours = total_time_seconds / 3600
        print(f"NWB file written in {write_time:.2f}s")
        print(f"PROCESSED NWB file size: {nwb_size_gb:.2f} GB ({nwb_size_bytes:,} bytes)")
        print(f"Write speed: {nwb_size_gb / (write_time / 3600):.2f} GB/hour")
        print(f"PROCESSED conversion total time: {total_time_seconds:.2f}s")
        print(f"PROCESSED conversion total time: {total_time_hours:.2f} hours")
        print(f"PROCESSED conversion completed: {nwbfile_path}")
        print(f"PROCESSED NWB saved to: {nwbfile_path}")

    return {
        "nwbfile_path": nwbfile_path,
        "nwb_size_bytes": nwb_size_bytes,
        "nwb_size_gb": nwb_size_gb,
        "write_time": write_time,
    }


if __name__ == "__main__":
    # Example usage
    convert_processed_session(
        eid="5ce2e17e-8471-42d4-8a16-21949710b328",
        one=ONE(),  # base_url="https://alyx.internationalbrainlab.org"
        stub_test=True,
        base_path=Path("E:/IBL-data-share"),
        append_on_disk_nwbfile=False,
        verbose=True,
    )
