"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import time
from pathlib import Path

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
from ndx_ibl import IblSubject
from neuroconv.utils import dict_deep_update, load_dict_from_file
from one.api import ONE
from pynwb import NWBFile

from ibl_mesoscope_to_nwb.mesoscope2025 import ProcessedMesoscopeNWBConverter
from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import (
    MesoscopeImageAnatomicalLocalizationInterface,
    MesoscopeMotionCorrectedImagingInterface,
    MesoscopeROIAnatomicalLocalizationInterface,
    MesoscopeSegmentationInterface,
    MesoscopeWheelKinematicsInterface,
    MesoscopeWheelMovementsInterface,
    MesoscopeWheelPositionInterface,
)
from ibl_mesoscope_to_nwb.mesoscope2025.utils import (
    get_available_tasks_from_alf_collections,
    get_FOV_names_from_alf_collections,
    sanitize_subject_id_for_dandi,
)


def convert_processed_session(
    eid: str,
    one: ONE,
    output_path: Path,
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
    output_path : Path, optional
        Base output directory for NWB files.
    append_on_disk_nwbfile: bool, optional
        If True, append to an existing on-disk NWB file instead of creating a new one.
    Returns
    -------
    dict
        Conversion result information including NWB file path and timing
    """
    if verbose:
        print(f"Starting PROCESSED conversion for session {eid}...")
    start_time = time.time()

    # Setup paths
    session_info = one.alyx.rest("sessions", "read", id=eid)
    subject_nickname = session_info.get("subject")
    if isinstance(subject_nickname, dict):
        subject_nickname = subject_nickname.get("nickname") or subject_nickname.get("name")
    if not subject_nickname:
        subject_nickname = "unknown"

    # Sanitize subject nickname for DANDI compliance (replace underscores with hyphens)
    subject_id_for_filenames = sanitize_subject_id_for_dandi(subject_nickname)

    # New structure: nwbfiles/{full|stub}/sub-{subject}/*.nwb
    conversion_mode = "stub" if stub_test else "full"
    output_dir = output_path / conversion_mode / f"sub-{subject_id_for_filenames}"
    output_dir.mkdir(parents=True, exist_ok=True)
    nwbfile_path = (
        output_dir
        / f"sub-{subject_id_for_filenames}"
        / f"sub-{subject_id_for_filenames}_ses-{eid}_desc-raw_behavior+ophys.nwb"
    )

    # ========================================================================
    # STEP 1: Define data interfaces
    # ========================================================================

    if verbose:
        print(f"Creating data interfaces...")
    interface_creation_start = time.time()

    data_interfaces = dict()
    conversion_options = dict()
    interface_kwargs = dict(one=one, session=eid)

    processed_FOVs = get_FOV_names_from_alf_collections(**interface_kwargs)
    processed_FOVs = processed_FOVs[:2] if stub_test else processed_FOVs  # Limit to first 2 planes for testing
    for FOV_index, FOV_name in enumerate(processed_FOVs):
        # Add Motion Corrected Imaging
        assert MesoscopeMotionCorrectedImagingInterface.check_availability(
            one, eid, FOV_name=FOV_name
        ), f"Motion Corrected Imaging data not available for FOV '{FOV_name}' in session {eid}"
        data_interfaces[f"{FOV_name}MotionCorrectedImaging"] = MesoscopeMotionCorrectedImagingInterface(
            **interface_kwargs, FOV_name=FOV_name, verbose=verbose
        )
        conversion_options.update(
            {f"{FOV_name}MotionCorrectedImaging": dict(stub_test=stub_test, photon_series_index=FOV_index)}
        )
        assert MesoscopeSegmentationInterface.check_availability(
            one, eid, FOV_name=FOV_name
        ), f"Segmentation data not available for FOV '{FOV_name}' in session {eid}"
        # Add Segmentation
        data_interfaces[f"{FOV_name}Segmentation"] = MesoscopeSegmentationInterface(
            **interface_kwargs, FOV_name=FOV_name, verbose=verbose
        )
        conversion_options.update({f"{FOV_name}Segmentation": dict(stub_test=stub_test)})

        # Add Anatomical Localization
        if MesoscopeROIAnatomicalLocalizationInterface.check_availability(one, eid, FOV_name=FOV_name)["available"]:
            data_interfaces[f"{FOV_name}ROIAnatomicalLocalization"] = MesoscopeROIAnatomicalLocalizationInterface(
                **interface_kwargs, FOV_name=FOV_name
            )
            conversion_options.update({f"{FOV_name}ROIAnatomicalLocalization": dict()})
        else:
            if verbose:
                print(f"  - ROI Anatomical Localization not available for FOV '{FOV_name}'")
        if MesoscopeImageAnatomicalLocalizationInterface.check_availability(one, eid, FOV_name=FOV_name)["available"]:
            data_interfaces[f"{FOV_name}ImageAnatomicalLocalization"] = MesoscopeImageAnatomicalLocalizationInterface(
                **interface_kwargs, FOV_name=FOV_name
            )
            conversion_options.update({f"{FOV_name}ImageAnatomicalLocalization": dict()})
        else:
            if verbose:
                print(f"  - Image Anatomical Localization not available for FOV '{FOV_name}'")

    # Behavioral data
    data_interfaces["BrainwideMapTrials"] = BrainwideMapTrialsInterface(**interface_kwargs)
    conversion_options.update({"BrainwideMapTrials": dict(stub_test=stub_test, stub_trials=10)})

    # Wheel data - add each interface if its data is available
    processed_tasks = get_available_tasks_from_alf_collections(**interface_kwargs)
    for task in processed_tasks:
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
        output_path=Path("E:/IBL-mesoscope-nwbfiles"),
        append_on_disk_nwbfile=False,
        verbose=True,
    )
