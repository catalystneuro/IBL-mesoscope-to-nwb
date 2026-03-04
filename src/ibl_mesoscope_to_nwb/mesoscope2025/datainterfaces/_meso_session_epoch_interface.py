"""
Interface for session-level epochs (high-level task vs passive phases).

This module provides an interface for adding simple session-level epochs to NWB files,
defining the two main phases: task/experiment phase and passive phase.
"""

import logging
import time
from datetime import datetime
from typing import Optional

from dateutil import tz
from ibl_to_nwb.datainterfaces._base_ibl_interface import BaseIBLDataInterface
from ibllib.io.session_params import get_task_collection, get_task_protocol
from one.api import ONE
from pynwb import NWBFile
from pynwb.epoch import TimeIntervals


class SessionEpochsInterface(BaseIBLDataInterface):
    """
    Interface for session-level epoch timing data.

    This interface handles the high-level epochs table that defines the two main
    phases of an IBL session: the task/experiment phase and the passive phase.
    """

    REVISION = None

    def __init__(
        self,
        one: ONE,
        session: str,
    ):
        """
        Initialize the session epochs interface.

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
        Declare data files required for session epochs.

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
                "standard": ["_iblrig_taskSettings.raw.json"],
            },
        }

    @classmethod
    def download_data(
        cls, one: ONE, eid: str, download_only: bool = True, logger: Optional[logging.Logger] = None, **kwargs
    ) -> dict:
        """
        Download session epochs data.

        NOTE: Uses the class-level REVISION attribute to determine which revision to download.

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

        # Find the last available revision from candidates
        revision = cls.REVISION

        if logger:
            logger.info(f"Downloading session epochs data (session {eid}, revision {revision})")

        start_time = time.time()

        # Download the intervals table
        # Note: Must separate collection and filename for ONE API
        one.load_dataset(
            eid,
            "_iblrig_taskSettings.raw.json",
            collection="alf",
            revision=revision,
            download_only=download_only,
        )

        download_time = time.time() - start_time

        if logger:
            logger.info(f"  Downloaded session epochs data in {download_time:.2f}s")

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
        Add session-level epochs to the NWB file.

        Creates two epochs defining:
        - Task/experiment phase (0 to start of passive period)
        - Passive phase (start to end of passive period)

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to add data to
        metadata : dict, optional
            Additional metadata (not currently used)
        """

        # Initialize epochs table if it doesn't exist
        if nwbfile.epochs is None:
            epochs_description = (
                "Session-level epochs defining the two main phases of an IBL recording session. "
                "The 'task' epoch covers the active behavioral task period where the mouse performs "
                "the decision-making task (responding to visual stimuli by turning a wheel). "
                "The 'passive' epoch covers the passive replay period where visual and auditory stimuli "
                "are presented without the mouse performing any task, used for receptive field mapping "
                "and stimulus response characterization. The passive protocol includes replay of task stimuli, "
                "sparse noise for receptive field mapping, and natural movie clips. "
                "See the 'protocol_type' column to distinguish between epochs."
            )
            nwbfile.epochs = TimeIntervals(name="epochs", description=epochs_description)

        # Add custom columns to the epochs table
        if "protocol_type" not in nwbfile.epochs.colnames:
            nwbfile.epochs.add_column(name="protocol_type", description="Type of protocol phase (task or passive)")

        if "epoch_description" not in nwbfile.epochs.colnames:
            nwbfile.epochs.add_column(
                name="epoch_description", description="Detailed description of what occurs during this epoch"
            )

        # Epoch descriptions
        task_description = (
            "Active behavioral task period. The mouse performs a decision-making task where it "
            "must turn a wheel to move a visual stimulus (Gabor patch) to the center of the screen. "
            "Correct responses are rewarded with water; incorrect responses trigger white noise feedback. "
            "The trials table contains detailed timing and outcome data for each trial during this epoch."
        )

        passive_description = (
            "Passive stimulus replay period. Visual and auditory stimuli are presented while the mouse "
            "is head-fixed but not performing any task. This epoch includes: (1) replay of task-relevant "
            "stimuli (Gabor patches at various contrasts and positions), (2) sparse noise stimuli for "
            "receptive field mapping, and (3) natural movie clips. Used for characterizing sensory responses "
            "independent of task engagement."
        )

        # Get session start time and lab timezone from Alyx (same pattern as VisualStimulusInterface)
        (session_metadata,) = self.one.alyx.rest(url="sessions", action="list", id=self.session)
        (lab_metadata,) = self.one.alyx.rest("labs", "list", name=session_metadata["lab"])
        session_start_time = datetime.fromisoformat(session_metadata["start_time"])
        tzinfo = tz.gettz(lab_metadata["timezone"])
        session_start_time = session_start_time.replace(tzinfo=tzinfo)

        # Load experiment description to iterate over protocols and their collections
        experiment_description = self.one.load_dataset(id=self.session, dataset="_ibl_experiment.description")
        protocols = get_task_protocol(experiment_description)

        if protocols is None:
            raise ValueError(
                f"No task protocols found in experiment description for session {self.session}. "
                "Cannot extract epoch timing without protocol definitions."
            )

        for protocol in protocols:
            collection = get_task_collection(experiment_description, protocol)

            # Load the task settings JSON for this protocol's collection
            task_settings = self.one.load_dataset(
                id=self.session,
                dataset="_iblrig_taskSettings.raw.json",
                collection=collection,
            )

            # Extract epoch start/stop times from task settings
            epoch_start_dt = datetime.fromisoformat(task_settings["SESSION_START_TIME"]).replace(tzinfo=tzinfo)
            epoch_end_dt = datetime.fromisoformat(task_settings["SESSION_END_TIME"]).replace(tzinfo=tzinfo)

            # Convert to seconds relative to the NWB session start time
            epoch_start = (epoch_start_dt - session_start_time).total_seconds()
            epoch_end = (epoch_end_dt - session_start_time).total_seconds()

            if "passive" in protocol:
                nwbfile.epochs.add_interval(
                    start_time=epoch_start,
                    stop_time=epoch_end,
                    protocol_type="passive",
                    epoch_description=passive_description,
                )
            else:
                nwbfile.epochs.add_interval(
                    start_time=epoch_start,
                    stop_time=epoch_end,
                    protocol_type="task",
                    epoch_description=task_description,
                )
