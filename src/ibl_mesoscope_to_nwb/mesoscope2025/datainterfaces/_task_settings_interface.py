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

PROTOCOLS_MAPPING = {
    "cuedBiasedChoiceWorld": {
        "protocol_type": "active",
        "protocol_description": (
            "Cued biased choice world — a custom variant of the biased choice world task with added visual cues. "
            "The mouse performs a decision-making task: a Gabor patch appears on the left or right of the screen, "
            "and the mouse turns a steering wheel to bring it to the center. Correct responses are rewarded with water. "
            "Stimulus probability alternates between 80/20 and 20/80 blocks. All contrast levels are used "
            "(100%, 25%, 12.5%, 6.25%, 0%)."
        ),
    },
    "biasedChoiceWorld": {
        "protocol_type": "active",
        "protocol_description": (
            "Biased choice world — the standard IBL data-collection task for trained mice. "
            "A Gabor patch appears at ±35° azimuth and the mouse turns a wheel to bring it to the center. "
            "Correct responses earn a water reward (~1.5 µL); incorrect responses trigger white noise and a 2s timeout. "
            "Stimulus probability alternates between 80/20 and 20/80 blocks (starting with a 50/50 block), "
            "with block lengths drawn from a truncated exponential distribution (min 20, max 100 trials). "
            "Full contrast set: [1.0, 0.25, 0.125, 0.0625, 0.0]. "
        ),
    },
    "advancedChoiceWorld": {
        "protocol_type": "active",
        "protocol_description": (
            "Advanced choice world — the iblrig v8+ replacement for biasedChoiceWorld. "
            "Functionally equivalent to biasedChoiceWorld (same biased block structure, contrast set, and trial logic) "
            "but with updated software architecture using Bonsai for visual stimulus rendering and an improved "
            "Bpod state machine integration."
        ),
    },
    "passiveChoiceWorld": {
        "protocol_type": "passive",
        "protocol_description": (
            "Passive choice world — replay of choice world stimuli without behavioral contingency. "
            "Gabor patches at all contrasts and positions, go cue tones, and white noise bursts are presented "
            "in randomized order while the mouse is head-fixed but not performing any task. "
            "No reward is delivered and no response is required. "
            "Used to compare neural responses during active decision-making vs. passive viewing, "
            "isolating sensory from decision- and movement-related signals."
        ),
    },
    "sparseNoise": {
        "protocol_type": "passive",
        "protocol_description": (
            "Sparse noise stimulus for receptive field mapping. "
            "Sparse white and black squares are presented at random screen locations to map the spatial "
            "receptive fields of visually responsive neurons. Each frame contains a small number of "
            "active squares on a gray background, allowing efficient estimation of spatial receptive fields."
        ),
    },
    "passiveVideo": {
        "protocol_type": "passive",
        "protocol_description": (
            "Passive video presentation — typically a Perlin noise video (spatio-temporal noise pattern) "
            "played to the mouse for retinotopic mapping and visual cortex characterization. "
            "No behavioral contingency — the mouse simply views the screen. "
        ),
    },
    "spontaneous": {
        "protocol_type": "passive",
        "protocol_description": (
            "Spontaneous activity recording — no stimuli or task. "
            "A gray screen is displayed while neural activity is recorded. "
            "Used to characterize baseline neural dynamics, ongoing activity, and resting-state patterns. "
            "Typically lasts 5-10 minutes."
        ),
    },
    "trainingChoiceWorld": {
        "protocol_type": "training",
        "protocol_description": (
            "Training choice world — the standard IBL training task. "
            "Mice learn to turn a wheel to move a Gabor patch to the center of the screen. "
            "Uses adaptive contrast levels: training starts with only high-contrast stimuli (100%) "
            "and progressively introduces harder contrasts (25%, 12.5%, 6.25%, 0%) as performance improves. "
            "No probability blocks — stimulus appears 50/50 left/right throughout. "
            "The mouse progresses through training stages until meeting criteria for the biased task."
        ),
    },
    "trainingPhaseChoiceWorld": {
        "protocol_type": "training",
        "protocol_description": (
            "Training phase choice world — the iblrig v8+ replacement for trainingChoiceWorld. "
            "Same adaptive training progression (phases 0-5) with Bonsai-based visual stimulus rendering. "
            "Phase transitions are based on performance metrics identical to trainingChoiceWorld criteria."
        ),
    },
    "habituationChoiceWorld": {
        "protocol_type": "habituation",
        "protocol_description": (
            "Habituation choice world — pre-training habituation task. "
            "Mice are exposed to the rig environment, visual stimuli, and reward delivery "
            "without requiring any wheel turns. The stimulus moves to the center automatically "
            "and water is delivered freely on every trial. Only high-contrast stimuli (100%) are used. "
            "Typically runs for 1-3 days before formal training begins."
        ),
    },
    "ephysChoiceWorld": {
        "protocol_type": "active",
        "protocol_description": (
            "Electrophysiology choice world — biasedChoiceWorld configured for sessions with "
            "simultaneous Neuropixels electrophysiology recordings. Behaviorally identical to biasedChoiceWorld "
            "but with additional synchronization signals for alignment with neural recordings."
            "A Gabor patch appears at ±35° azimuth and the mouse turns a wheel to bring it to the center. "
            "Correct responses earn a water reward (~1.5 µL); incorrect responses trigger white noise and a 2s timeout. "
            "Stimulus probability alternates between 80/20 and 20/80 blocks (starting with a 50/50 block), "
            "with block lengths drawn from a truncated exponential distribution (min 20, max 100 trials). "
            "Full contrast set: [1.0, 0.25, 0.125, 0.0625, 0.0]. "
        ),
    },
    "tonotopicMapping": {
        "protocol_type": "passive",
        "protocol_description": (
            "Tonotopic mapping — presents auditory stimuli at different frequencies "
            "to map the tonotopic organization of auditory cortex. "
            "Used alongside mesoscope imaging for characterizing auditory responses."
        ),
    },
}


class TaskSettingsInterface(BaseIBLDataInterface):
    """
    Interface for session-level epoch timing data.

    This interface handles the high-level epochs table that defines the two main
    phases of an IBL session: the task/experiment phase and the passive phase.
    TODO add more detailed description
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
                "The 'habituation' and 'training' epochs cover the respective periods during which the mouse undergoes habituation "
                "and training for the task. "
                "See the 'protocol_type' column to distinguish between epochs."
            )
            nwbfile.epochs = TimeIntervals(name="epochs", description=epochs_description)

        # Add custom columns to the epochs table
        if "protocol_type" not in nwbfile.epochs.colnames:
            nwbfile.epochs.add_column(name="protocol_type", description="Type of protocol phase (task or passive)")

        if "protocol_description" not in nwbfile.epochs.colnames:
            nwbfile.epochs.add_column(
                name="protocol_description", description="Detailed description of what occurs during this task protocol"
            )

        if "task_settings" not in nwbfile.epochs.colnames:
            nwbfile.epochs.add_column(
                name="task_settings",
                description="Settings and parameters for the task protocol (_iblrig_taskSettings.raw.json content as a dictionary)",
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

            # Determine protocol type and description from the mapping
            # NB: the actual protocol name in the experiment description may have additional suffixes (e.g. "cuedBiasedChoiceWorld_1"), so we check for substrings
            protocol_type = None
            protocol_description = None
            for key, value in PROTOCOLS_MAPPING.items():
                if key in protocol:
                    protocol_type = value["protocol_type"]
                    protocol_description = value["protocol_description"]
                    break

            assert (
                protocol_type is not None
            ), f"Protocol '{protocol}' not found in known protocols: {list(PROTOCOLS_MAPPING.keys())}. Cannot determine protocol type and description for epochs."

            nwbfile.epochs.add_interval(
                start_time=epoch_start,
                stop_time=epoch_end,
                protocol_type=protocol_type,
                protocol_description=protocol_description,
                task_settings=str(task_settings),
            )
