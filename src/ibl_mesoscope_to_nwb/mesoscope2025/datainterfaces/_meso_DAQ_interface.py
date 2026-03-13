import warnings
from pathlib import Path

import numpy as np
from ibl_to_nwb.datainterfaces._base_ibl_interface import BaseIBLDataInterface
from ibllib.io.raw_daq_loaders import timeline_meta2chmap, timeline_meta2wiring
from neuroconv.utils import dict_deep_update, load_dict_from_file
from one.api import ONE
from pynwb import NWBFile
from pynwb.base import TimeSeries

# =============================================================================
# Digital Device Labels (needed at init time for digital_channel_groups)
# These define how to interpret polarities values (0/1) for each device type
# =============================================================================

DIGITAL_DEVICE_LABELS = {
    "neural_frames": {0: "frame_low", 1: "frame_high"},
    "volume_counter": {0: "volume_low", 1: "volume_high"},
    "bpod": {0: "bpod_low", 1: "bpod_high"},
    "frame2ttl": {0: "screen_dark", 1: "screen_bright"},
    "left_camera": {0: "exposure_end", 1: "frame_start"},
    "right_camera": {0: "exposure_end", 1: "frame_start"},
    "body_camera": {0: "exposure_end", 1: "frame_start"},
    "audio": {0: "audio_off", 1: "audio_on"},
    "rotary_encoder": {0: "phase_low", 1: "phase_high"},
}


class MesoscopeDAQInterface(BaseIBLDataInterface):
    """
    IBL-specific DAQ interface that uses _timeline_DAQdata.meta.json for channel configuration.

    This interface:
    1. Build digital_channel_groups and analog_channel_groups from _timeline_DAQdata.meta.json
    2. Load static NWB metadata (name, description, meanings) from YAML

    The _timeline_DAQdata.meta.json file documents how behavioral devices are connected to sync
    channels and varies by rig, making it essential session-specific metadata.

    Example _timeline_DAQdata.meta.json structure:
    {
        "SYSTEM": "timeline",
        "SYNC_WIRING_ANALOG": {
            'ai0': 'chrono',
            'ai3': 'photoDiode',
            'ai4': 'GalvoX',
            'ai5': 'GalvoY',
            'ai8': 'RemoteFocus1',
            'ai9': 'RemoteFocus2',
            'ai6': 'LaserPower',
            'ai7': 'bpod',
            'ai10': 'reward_valve',
            'ai11': 'frame2ttl',
            'ai12': 'left_camera',
            'ai13': 'right_camera',
            'ai14': 'body_camera',
            'ai15': 'audio',
            'ai1': 'syncEcho',
            'ai2': 'photosensor'
        },
        "SYNC_WIRING_DIGITAL": {
            'ctr0': 'neural_frames',
            'ctr2': 'volume_counter',
            'ctr3': 'rotary_encoder',
            'port0/line2': 'acqLive',
        }
    }

    Some analog channels are stored as digital channels in the wiring (e.g. bpod, rotary_encoder) because they have digital-like behavior (polarity changes). These are included in digital_channel_groups with appropriate labels_map.
    """

    display_name = "Mesoscope DAQ"
    info = "Interface for Mesoscope DAQ board recording data."
    interface_name = "MesoscopeDAQInterface"
    REVISION: str | None = None

    @classmethod
    def get_data_requirements(cls, **kwargs) -> dict:
        """
        Get data requirements for DAQ interface.

        Returns
        -------
        dict
            Dictionary with required DAQ files.
        """
        return {
            "exact_files_options": {
                "standard": [
                    "raw_sync_data/_timeline_DAQdata.raw.npy",
                    "raw_sync_data/_timeline_DAQdata.timestamps.npy",
                    "raw_sync_data/_timeline_DAQdata.meta.json",
                ],
            },
        }

    def __init__(self, one: ONE, session: str, metadata_key: str = "IblDAQ"):
        """Initialize the MesoscopeDAQInterface.

        Args:
            one (ONE): An instance of the ONE API.
            session (str): The session ID.
        """
        self.one = one
        self.session = session
        self.revision = self.REVISION
        self.metadata_key = metadata_key

        self.raw_metadata = self.one.load_dataset(
            self.session, dataset="_timeline_DAQdata.meta.json", collection="raw_sync_data"
        )

        session_path = one.eid2path(session)
        # Load _timeline_DAQdata.meta.json which maps hardware ports to device names
        self.wiring = timeline_meta2wiring(Path(session_path) / "raw_sync_data")

        # Build channel groups from wiring
        self._digital_channel_groups = self.get_digital_channel_groups_from_wiring(self.wiring)
        self._analog_channel_groups = self.get_analog_channel_groups_from_wiring(self.wiring)

        if len(self._digital_channel_groups) > 0:
            self.has_digital_channels = True

        if len(self._analog_channel_groups) > 0:
            self.has_analog_channels = True

    @staticmethod
    def get_digital_channel_groups_from_wiring(wiring: dict) -> dict:
        """
        Build digital_channel_groups from _timeline_DAQdata.meta.json.

        Maps each digital device in _timeline_DAQdata.meta.json to its channel ID and labels_map.

        Parameters
        ----------
        wiring : dict
            Wiring configuration loaded from _timeline_DAQdata.meta.json

        Returns
        -------
        dict
            digital_channel_groups structure.
            Example: {
                "left_camera": {
                    "channels": {
                        "ctr0": {"labels_map": {0: "exposure_end", 1: "frame_start"}}
                    }
                }
            }
        """
        digital_channel_groups = {}
        digital_wiring = wiring.get("SYNC_WIRING_DIGITAL", {})

        for digital_input, device_name in digital_wiring.items():
            if digital_input.startswith("ctr"):
                channel_id = f"daq#XD{digital_input.replace('ctr', '')}"
                if device_name in DIGITAL_DEVICE_LABELS:
                    digital_channel_groups[device_name] = {
                        "channels": {channel_id: {"labels_map": DIGITAL_DEVICE_LABELS[device_name]}}
                    }

        # TODO: Why some analog channels are also included in digital_channel_groups? (e.g. bpod, rotary_encoder)
        analog_wiring = wiring.get("SYNC_WIRING_ANALOG", {})
        for analog_input, device_name in analog_wiring.items():
            if device_name in DIGITAL_DEVICE_LABELS:
                channel_id = f"daq#XA{analog_input.replace('ai', '')}"
                digital_channel_groups[device_name] = {
                    "channels": {channel_id: {"labels_map": DIGITAL_DEVICE_LABELS[device_name]}}
                }

        return digital_channel_groups

    @staticmethod
    def get_analog_channel_groups_from_wiring(wiring: dict) -> dict:
        """
        Build analog_channel_groups from _timeline_DAQdata.meta.json.

        Maps each analog device in _timeline_DAQdata.meta.json to its channel ID.
        Excludes laser-related channels which are not part of Mesoscope conversion.

        Parameters
        ----------
        wiring : dict
            Wiring configuration loaded from _timeline_DAQdata.meta.json

        Returns
        -------
        dict
            analog_channel_groups structure.
            Example: {"bpod": {"channels": ["daq#XA7"]}}
        """
        # TODO: Should we exclude devices that are already included as digital channels?
        excluded_devices = DIGITAL_DEVICE_LABELS

        analog_channel_groups = {}
        analog_wiring = wiring.get("SYNC_WIRING_ANALOG", {})

        for analog_input, device_name in analog_wiring.items():
            if device_name in excluded_devices:
                continue
            if analog_input.startswith("ai"):
                channel_id = f"daq#XA{analog_input.replace('ai', '')}"
                analog_channel_groups[device_name] = {"channels": [channel_id]}

        return analog_channel_groups

    def get_sampling_frequency(self) -> float:
        return 1 / self.raw_metadata.get("samplingInterval")

    def get_starting_time(self) -> float:
        """Get the starting time of the recording. The _timeline_DAQdata.timestamps.npy map the first and last samples with the relative timestamps expressed in seconds.
        E.g. timestamps =
                array([[      0.   ,       0.   ],
                       [4018999.   ,    4018.999]])

         Returns:
             float: The starting time in seconds
        """
        timestamps = self.one.load_dataset(
            self.session, dataset="_timeline_DAQdata.timestamps.npy", collection="raw_sync_data"
        )
        return timestamps[0][1]

    def _get_channel_index(self, channel_name: str) -> int:
        """
        Get the index of a specific channel.

        Parameters
        ----------
        channel_name : str
            The name of the channel.

        Returns
        -------
        int
            The index of the specified channel in the DAQ recording.
        """
        chmap = timeline_meta2chmap(self.raw_metadata, include_channels=[channel_name])
        return chmap[channel_name] - 1  # Convert to 0-based index

    def get_metadata(self):
        """
        Get metadata with IBL-specific channel configurations.

        Loads static metadata from YAML and filters to only include devices
        present in this session's _timeline_DAQdata.meta.json.

        Returns
        -------
        dict
            Metadata dictionary with:
            - Events metadata for digital channels (name, description, meanings)
            - TimeSeries metadata for analog channels (name, description)
        """
        metadata = super().get_metadata()

        # Load static metadata from YAML
        static_metadata = load_dict_from_file(
            file_path=Path(__file__).parent.parent / "_metadata" / "mesoscope_DAQ_metadata.yaml"
        )

        # Get devices present in this session's wiring
        analog_devices = set(self.wiring.get("SYNC_WIRING_ANALOG", {}).values())
        digital_devices = set(self.wiring.get("SYNC_WIRING_DIGITAL", {}).values()) | (
            set(self.wiring.get("SYNC_WIRING_ANALOG", {}).values()) & set(DIGITAL_DEVICE_LABELS.keys())
        )

        # Filter TimeSeries metadata to only include devices in wiring
        timeseries_metadata = {}
        for device in analog_devices:
            if device in static_metadata.get("TimeSeries", {}):
                timeseries_metadata[device] = static_metadata["TimeSeries"][device].copy()
            else:
                warnings.warn(
                    f"No metadata configured for analog device '{device}'. "
                    f"Add an entry to _metadata/mesoscope_DAQ_metadata.yml.",
                    UserWarning,
                    stacklevel=2,
                )

        if timeseries_metadata:
            metadata = dict_deep_update(metadata, {"TimeSeries": {self.metadata_key: timeseries_metadata}})

        # Filter Events metadata to only include devices in wiring
        events_metadata = {}
        for device in digital_devices:
            if device in static_metadata.get("Events", {}):
                events_metadata[device] = static_metadata["Events"][device].copy()
            else:
                warnings.warn(
                    f"No metadata configured for digital device '{device}'. "
                    f"Add an entry to _metadata/mesoscope_DAQ_metadata.yml.",
                    UserWarning,
                    stacklevel=2,
                )

        if events_metadata:
            metadata = dict_deep_update(metadata, {"Events": {self.metadata_key: events_metadata}})

        return metadata

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict | None = None,
        *,
        stub_test: bool = False,
    ):
        """
        Add DAQ board data to an NWB file, including both analog and digital channels if present.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to which the DAQ data will be added
        metadata : dict | None, default: None
            Metadata dictionary with device information. If None, uses default metadata
        stub_test : bool, default: False
            If True, only writes a small amount of data for testing
        """
        metadata = metadata or self.get_metadata()
        timeline = self.one.load_object(self.session, "DAQdata")

        if stub_test:
            data = timeline["raw"][:10000, :]
            # timestamps = timeline["timestamps"][:10000, :]

        # Add devices
        device_metadata = metadata.get("Devices", [])
        for device in device_metadata:
            if device["name"] not in nwbfile.devices:
                nwbfile.create_device(**device)

        # Add analog and digital channels
        if self.has_analog_channels:
            self._add_analog_channels(
                nwbfile=nwbfile,
                data=data,
                metadata=metadata,
            )

        if self.has_digital_channels:
            self._add_digital_channels(nwbfile=nwbfile, metadata=metadata, timeline_object=timeline)

    def _add_analog_channels(
        self,
        nwbfile: NWBFile,
        data: np.ndarray,
        metadata: dict,
    ):
        """
        Add analog channels from the DAQ board to the NWB file.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to add the analog channels to
        data : np.ndarray
            The raw DAQ data array loaded from _timeline_DAQdata.raw.npy
        metadata : dict
            Metadata dictionary with TimeSeries information
        """
        if not self.has_analog_channels:
            return
        # Get TimeSeries configurations from metadata
        time_series_metadata = metadata.get("TimeSeries", {}).get(self.metadata_key, {})

        # Write each group as a TimeSeries
        for group_key, _ in self._analog_channel_groups.items():
            # Check if this group has metadata
            if group_key not in time_series_metadata:
                continue
            # Get metadata for this group
            ts_metadata = time_series_metadata[group_key]
            # Build TimeSeries kwargs from recording properties
            tseries_kwargs = {}
            tseries_kwargs["data"] = data[:, self._get_channel_index(group_key)]

            # Add starting time and rate
            tseries_kwargs["starting_time"] = self.get_starting_time()
            tseries_kwargs["rate"] = self.get_sampling_frequency()

            # Update with user-provided metadata
            tseries_kwargs.update(ts_metadata)

            # Create TimeSeries object and add it to nwbfile
            time_series = TimeSeries(**tseries_kwargs)
            nwbfile.add_acquisition(time_series)

    def _add_digital_channels(
        self,
        nwbfile: NWBFile,
        metadata: dict,
        timeline_object: object | None = None,
    ):
        """
        Add digital channels from the DAQ board to the NWB file as events.

        Data structure (which channels, labels_map) comes from channel groups config.
        NWB properties (name, description, meanings) come from metadata.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to add the digital channels to
        metadata : dict
            Metadata dictionary containing channel configurations.
        """
        if not self.has_digital_channels:
            return

        from ibllib.io.raw_daq_loaders import extract_sync_timeline
        from ndx_events import LabeledEvents

        events_metadata = metadata.get("Events", {}).get(self.metadata_key, {})

        for group_key, group_config in self._digital_channel_groups.items():
            channels_config = group_config["channels"]
            # Get the single channel (validated at init to be single-channel for user groups)
            _, channel_config = next(iter(channels_config.items()))

            # Get labels_map from config (data structure)
            labels_map = channel_config["labels_map"]

            # Get NWB properties from metadata
            group_metadata = events_metadata.get(group_key, {})
            name = group_metadata.get("name")
            description = group_metadata.get("description")

            # Append meanings to description if provided
            # Future: when ndx-events MeaningsTable is integrated into NWB core,
            # these will be written to MeaningsTable instead of the description
            meanings = group_metadata.get("meanings", {})
            if meanings:
                meanings_text = "\n".join(f"  - {label}: {meaning}" for label, meaning in meanings.items())
                description = f"{description}\n\nLabel meanings:\n{meanings_text}"

            # Get event data
            chmap = timeline_meta2chmap(self.raw_metadata, include_channels=[group_key])
            events_structure = extract_sync_timeline(timeline_object, chmap=chmap)

            timestamps = events_structure["times"]
            data = (events_structure["polarities"] == 1).astype(int)  # Convert polarities from (-1, 1) to (0, 1)

            if timestamps.size == 0:
                continue

            # Build labels list from labels_map
            sorted_items = sorted(labels_map.items())
            labels_list = [label for _, label in sorted_items]

            labeled_events = LabeledEvents(
                name=name,
                description=description,
                timestamps=timestamps,
                data=data,
                labels=labels_list,
            )
            nwbfile.add_acquisition(labeled_events)
