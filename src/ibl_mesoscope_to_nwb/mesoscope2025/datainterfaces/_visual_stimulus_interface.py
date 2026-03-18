"""
Interface for passive period intervals (detailed passive phase timing).

This module provides an interface for adding detailed passive protocol interval timing to NWB files,
defining when different phases of the passive period occur (spontaneous activity, RFM, task replay).
The intervals are stored in a custom TimeIntervals table in the processing module.
"""

import logging
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from dateutil import tz
from ibl_to_nwb.datainterfaces._base_ibl_interface import BaseIBLDataInterface
from ibllib.io.session_params import get_task_collection, get_task_protocol
from one.api import ONE
from pynwb import NWBFile
from pynwb.epoch import TimeIntervals
from pynwb.image import OpticalSeries


class VisualStimulusInterface(BaseIBLDataInterface):
    """
    Interface for adding visual stimulus information from passive protocol sessions to NWB files.
    """

    REVISION = None

    @staticmethod
    def find_passiveVideo_collection(one: ONE, session: str) -> Optional[str]:
        """
        Find the collection name for passiveVideo protocol data in the session metadata.

        Parameters
        ----------
        one : ONE
            ONE API instance for data access
        session : str
            Session ID (eid)

        Returns
        -------
        Optional[str]
            Collection name if found, otherwise None
        """
        metadata = one.load_dataset(id=session, dataset="_ibl_experiment.description")
        protocols = get_task_protocol(metadata)
        for protocol in protocols:
            if "passiveVideo" in protocol:
                collection = get_task_collection(metadata, protocol)
                if isinstance(collection, set):
                    raise NotImplementedError(
                        f"Multiple collections found for passiveVideo protocol: {collection}. "
                        "Automatic collection selection is not implemented."
                    )
                return collection
        return None

    def __init__(
        self,
        one: ONE,
        session: str,
    ):
        """
        Initialize the passive intervals interface.

        Parameters
        ----------
        one : ONE
            ONE API instance for data access
        session : str
            Session ID (eid)
        """
        super().__init__()
        self.one = one
        self.session = session
        self.revision = self.REVISION

    @classmethod
    def get_data_requirements(cls, **kwargs) -> dict:
        """
        Declare data files required for passive period intervals.

        Parameters
        ----------
        **kwargs
            Accepts but ignores kwargs for API consistency with base class.

        Returns
        -------
        dict
            Data requirements with exact file patterns
        """
        return {
            "exact_files_options": {
                "standard": ["_sp_taskData.raw.pqt", "_sp_video.mp4"],
            },
        }

    @classmethod
    def download_data(
        cls, one: ONE, eid: str, download_only: bool = True, logger: Optional[logging.Logger] = None, **kwargs
    ) -> dict:
        """
        Download passive period intervals data.

        NOTE: Queries ONE API to find the last available revision from REVISION_CANDIDATES.

        Parameters
        ----------
        one : ONE
            ONE API instance
        eid : str
            Session ID
        download_only : bool, default=True
            If True, download but don't load into memory
        logger : logging.Logger, optional
            Logger for progress tracking

        Returns
        -------
        dict
            Download status
        """
        requirements = cls.get_data_requirements()

        revision = cls.REVISION

        if logger:
            logger.info(f"Downloading passive intervals data (session {eid}, revision {revision})")

        start_time = time.time()

        # Download the intervals table
        collection = cls.find_passiveVideo_collection(one, eid)  # Check if collection exists and raise error if not

        one.load_collection(eid, collection=collection, object=["_sp_taskData", "_sp_video"], download_only=True)

        download_time = time.time() - start_time

        if logger:
            logger.info(f"  Downloaded passive intervals in {download_time:.2f}s")

        return {
            "success": True,
            "downloaded_objects": [],
            "downloaded_files": requirements["exact_files_options"]["standard"],
            "already_cached": [],
            "alternative_used": None,
            "data": None,
        }

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: Optional[dict] = None):
        """
        Add visual stimulus template and intervals to NWB file.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to add data to
        metadata : dict, optional
            Additional metadata (not currently used)
        """

        collection = self.find_passiveVideo_collection(
            self.one, self.session
        )  # Check if collection exists and raise error if not

        video_file_path = self.one.load_object(
            id=self.session, obj="_sp_video", collection=collection, revision=self.revision
        )["raw"]

        cap = cv2.VideoCapture(str(video_file_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()

        optical_series = OpticalSeries(
            name="visual_stimulus_video",
            description="Video played as visual stimulus during passive protocol",
            distance=1.0,  # TODO update placeholder value;
            unit="n.a.",
            external_file=[video_file_path],
            format="external",
            rate=fps,
            starting_time=np.nan,  # Placeholder value; actual timing of stimulus presentation is captured in the intervals table
        )
        nwbfile.add_stimulus_template(optical_series)

        # Create a custom TimeIntervals table for visual stimulus protocol intervals

        visual_stimulus_df = self.one.load_object(
            id=self.session, obj="_sp_taskData", collection=collection, revision=self.revision
        )["raw"]

        (session_metadata,) = self.one.alyx.rest(url="sessions", action="list", id=self.session)
        (lab_metadata,) = self.one.alyx.rest("labs", "list", name=session_metadata["lab"])
        session_start_time = datetime.fromisoformat(session_metadata["start_time"])
        tzinfo = tz.gettz(lab_metadata["timezone"])
        session_start_time = session_start_time.replace(tzinfo=tzinfo)

        visual_stimulus_intervals = TimeIntervals(
            name="visual_stimulus_intervals",
            description=f"Intervals for visual stimulus delivery during passive protocol.",
        )

        # Add custom column for protocol name
        visual_stimulus_intervals.add_column(
            name="visual_stimulus", description="Link to the video stimulus file played during this interval"
        )

        for start, stop in zip(visual_stimulus_df["intervals_0"], visual_stimulus_df["intervals_1"]):
            start_time = (datetime.fromtimestamp(start, tz=tzinfo) - session_start_time).total_seconds()
            stop_time = (datetime.fromtimestamp(stop, tz=tzinfo) - session_start_time).total_seconds()
            visual_stimulus_intervals.add_interval(
                start_time=start_time,
                stop_time=stop_time,
                visual_stimulus=optical_series,
            )

        nwbfile.add_stimulus(visual_stimulus_intervals)
