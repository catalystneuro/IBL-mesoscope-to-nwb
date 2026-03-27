"""Download functions for IBL mesoscope session data."""

import time

from ibl_to_nwb.datainterfaces import (
    BrainwideMapTrialsInterface,
    IblPoseEstimationInterface,
    LickInterface,
    PassiveIntervalsInterface,
    PassiveReplayStimInterface,
    PupilTrackingInterface,
    RawVideoInterface,
    RoiMotionEnergyInterface,
)
from one.api import ONE

from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import (
    MesoscopeDAQInterface,
    MesoscopeImageAnatomicalLocalizationInterface,
    MesoscopeMotionCorrectedImagingInterface,
    MesoscopeRawImagingInterface,
    MesoscopeROIAnatomicalLocalizationInterface,
    MesoscopeSegmentationInterface,
    TaskSettingsInterface,
    VisualStimulusInterface,
)
from ibl_mesoscope_to_nwb.mesoscope2025.utils import get_FOV_names_from_alf_collections


def download_processed_session_data(
    eid: str,
    one: ONE,
    stub_test: bool = False,
    verbose: bool = False,
) -> dict:
    """Download all processed session data required for conversion to NWB.

    Calls the ``download_data`` classmethod for each interface that has one.
    Interfaces without a dedicated ``download_data`` method (e.g. wheel interfaces)
    rely on ONE's caching when their constructor is called during conversion.

    Parameters
    ----------
    eid : str
        Experiment ID (session UUID).
    one : ONE
        ONE API instance.
    stub_test : bool, optional
        If True, limit FOV downloads to the first 2 planes for lightweight testing,
        by default False.
    verbose : bool, optional
        If True, print download progress, by default False.

    Returns
    -------
    dict
        Download statistics including elapsed time and number of interfaces downloaded.
    """
    if verbose:
        print(f"Downloading processed session data for {eid}...")
    download_start = time.time()

    interface_kwargs = dict(one=one, eid=eid)
    interfaces_downloaded = 0

    # -------------------------------------------------------------------------
    # Per-FOV imaging data
    # -------------------------------------------------------------------------
    FOV_names = get_FOV_names_from_alf_collections(one=one, session=eid)
    if stub_test:
        FOV_names = FOV_names[:2]

    for FOV_name in FOV_names:
        if verbose:
            print(f"  Downloading motion-corrected imaging for {FOV_name}...")
        MesoscopeMotionCorrectedImagingInterface.download_data(**interface_kwargs, FOV_name=FOV_name)
        interfaces_downloaded += 1

        if verbose:
            print(f"  Downloading segmentation for {FOV_name}...")
        MesoscopeSegmentationInterface.download_data(**interface_kwargs, FOV_name=FOV_name)
        interfaces_downloaded += 1

        if MesoscopeROIAnatomicalLocalizationInterface.check_availability(one, eid, FOV_name=FOV_name)["available"]:
            if verbose:
                print(f"  Downloading ROI anatomical localization for {FOV_name}...")
            MesoscopeROIAnatomicalLocalizationInterface.download_data(**interface_kwargs, FOV_name=FOV_name)
            interfaces_downloaded += 1

        if MesoscopeImageAnatomicalLocalizationInterface.check_availability(one, eid, FOV_name=FOV_name)["available"]:
            if verbose:
                print(f"  Downloading image anatomical localization for {FOV_name}...")
            MesoscopeImageAnatomicalLocalizationInterface.download_data(**interface_kwargs, FOV_name=FOV_name)
            interfaces_downloaded += 1

    # -------------------------------------------------------------------------
    # Behavioral data
    # -------------------------------------------------------------------------
    if verbose:
        print("  Downloading trials data...")
    BrainwideMapTrialsInterface.download_data(**interface_kwargs)
    interfaces_downloaded += 1

    if verbose:
        print("  Downloading task settings...")
    TaskSettingsInterface.download_data(**interface_kwargs)
    interfaces_downloaded += 1

    # Passive period interfaces (optional)
    if PassiveIntervalsInterface.check_availability(one, eid)["available"]:
        if verbose:
            print("  Downloading passive intervals...")
        PassiveIntervalsInterface.download_data(**interface_kwargs)
        interfaces_downloaded += 1

    if PassiveReplayStimInterface.check_availability(one, eid)["available"]:
        if verbose:
            print("  Downloading passive replay stimuli...")
        PassiveReplayStimInterface.download_data(**interface_kwargs)
        interfaces_downloaded += 1

    # Licks (optional)
    if LickInterface.check_availability(one, eid)["available"]:
        if verbose:
            print("  Downloading lick data...")
        LickInterface.download_data(**interface_kwargs)
        interfaces_downloaded += 1

    # -------------------------------------------------------------------------
    # Camera-based interfaces (pose estimation, pupil tracking, motion energy)
    # -------------------------------------------------------------------------
    for camera_view in ["left", "right", "body"]:
        camera_name = f"{camera_view}Camera"

        pose_availability = IblPoseEstimationInterface.check_availability(one, eid, camera_name=camera_name)
        if pose_availability["available"]:
            if verbose:
                print(f"  Downloading pose estimation for {camera_name}...")
            IblPoseEstimationInterface.download_data(**interface_kwargs, camera_name=camera_name)
            interfaces_downloaded += 1

        if camera_view in ["left", "right"]:
            if PupilTrackingInterface.check_availability(one, eid, camera_name=camera_name)["available"]:
                if verbose:
                    print(f"  Downloading pupil tracking for {camera_name}...")
                PupilTrackingInterface.download_data(**interface_kwargs, camera_name=camera_name)
                interfaces_downloaded += 1

        if RoiMotionEnergyInterface.check_availability(one, eid, camera_name=camera_name)["available"]:
            if verbose:
                print(f"  Downloading ROI motion energy for {camera_name}...")
            RoiMotionEnergyInterface.download_data(**interface_kwargs, camera_name=camera_name)
            interfaces_downloaded += 1

    download_time = time.time() - download_start
    if verbose:
        print(f"Processed data download completed in {download_time:.2f}s ({interfaces_downloaded} interfaces)")

    return {
        "download_time": download_time,
        "num_interfaces": interfaces_downloaded,
    }


def download_raw_session_data(
    eid: str,
    one: ONE,
    stub_test: bool = False,
    decompress: bool = True,
    verbose: bool = False,
) -> dict:
    """Download all raw session data required for conversion to NWB.

    Downloads raw imaging frames (with optional decompression), DAQ timeline,
    task settings, visual stimulus data, and behavioral videos.

    Parameters
    ----------
    eid : str
        Experiment ID (session UUID).
    one : ONE
        ONE API instance.
    stub_test : bool, optional
        If True, skip large video downloads not already in cache, by default False.
    decompress : bool, optional
        If True, decompress ``imaging.frames.tar.bz2`` archives after downloading,
        by default True.
    verbose : bool, optional
        If True, print download progress, by default False.

    Returns
    -------
    dict
        Download statistics including elapsed time and number of interfaces downloaded.
    """
    if verbose:
        print(f"Downloading raw session data for {eid}...")
    download_start = time.time()

    interface_kwargs = dict(one=one, eid=eid)
    interfaces_downloaded = 0

    # -------------------------------------------------------------------------
    # Raw imaging data (large tar.bz2 archives — downloads + decompresses)
    # -------------------------------------------------------------------------
    if verbose:
        decompress_note = " (with decompression)" if decompress else ""
        print(f"  Downloading raw imaging frames{decompress_note}...")
    MesoscopeRawImagingInterface.download_data(**interface_kwargs, decompress=decompress, verbose=verbose)
    interfaces_downloaded += 1

    # -------------------------------------------------------------------------
    # DAQ / timeline sync data
    # -------------------------------------------------------------------------
    if verbose:
        print("  Downloading DAQ timeline data...")
    MesoscopeDAQInterface.download_data(**interface_kwargs, verbose=verbose)
    interfaces_downloaded += 1

    # -------------------------------------------------------------------------
    # Task settings
    # -------------------------------------------------------------------------
    if verbose:
        print("  Downloading task settings...")
    TaskSettingsInterface.download_data(**interface_kwargs, verbose=verbose)
    interfaces_downloaded += 1

    # -------------------------------------------------------------------------
    # Visual stimulus data (passive video + task data)
    # -------------------------------------------------------------------------
    if verbose:
        print("  Downloading visual stimulus data...")
    VisualStimulusInterface.download_data(**interface_kwargs, verbose=verbose)
    interfaces_downloaded += 1

    # -------------------------------------------------------------------------
    # Behavioral videos
    # -------------------------------------------------------------------------
    for camera_view in ["left", "right", "body"]:
        if stub_test:
            # In stub mode, only download videos already in cache to avoid large transfers
            session_path = one.eid2path(eid)
            if session_path is None:
                if verbose:
                    print(f"  Stub mode: {camera_view}Camera video not in cache — skipping")
                continue
            video_path = session_path / f"raw_video_data/_iblrig_{camera_view}Camera.raw.mp4"
            if not video_path.exists():
                if verbose:
                    print(f"  Stub mode: {camera_view}Camera video not in cache — skipping")
                continue
            if verbose:
                print(f"  Stub mode: {camera_view}Camera already in cache — skipping download")
            continue

        if RawVideoInterface.check_availability(one, eid, camera_name=camera_view)["available"]:
            if verbose:
                print(f"  Downloading {camera_view}Camera video...")
            RawVideoInterface.download_data(**interface_kwargs, camera_name=camera_view, verbose=verbose)
            interfaces_downloaded += 1

    download_time = time.time() - download_start
    if verbose:
        print(f"Raw data download completed in {download_time:.2f}s ({interfaces_downloaded} interfaces)")

    return {
        "download_time": download_time,
        "num_interfaces": interfaces_downloaded,
    }
