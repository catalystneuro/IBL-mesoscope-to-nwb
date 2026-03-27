from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from dateutil import tz
from neuroconv import ConverterPipe
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.utils import dict_deep_update, load_dict_from_file
from one.api import ONE
from typing_extensions import Self

from ibl_mesoscope_to_nwb.mesoscope2025.utils import (
    get_ibl_subject_metadata,
    get_protocol_type_and_description,
)


class IblConverter(ConverterPipe):
    """Base NWB converter for IBL sessions.

    Fetches session-level metadata (start time, timezone, lab, institution, protocol,
    subject) from the Alyx REST API and merges it into the NWB metadata dict.

    Parameters
    ----------
    one : ONE
        ONE API instance for accessing IBL data.
    session : str
        Session ID (experiment UUID / eid).
    data_interfaces : list[BaseDataInterface] | dict[str, BaseDataInterface]
        Data interfaces to include in the conversion.
    verbose : bool, default=False
        If True, print progress messages during conversion.
    """

    def __init__(
        self,
        one: ONE,
        session: str,
        data_interfaces: list[BaseDataInterface] | dict[str, BaseDataInterface],
        verbose=False,
    ) -> Self:
        """Initialize the IBL converter.

        Parameters
        ----------
        one : ONE
            ONE API instance for accessing IBL data.
        session : str
            Session ID (experiment UUID / eid).
        data_interfaces : list[BaseDataInterface] | dict[str, BaseDataInterface]
            Data interfaces to include in the conversion.
        verbose : bool, default=False
            If True, print progress messages during conversion.
        """
        self.one = one
        self.session = session
        super().__init__(data_interfaces=data_interfaces, verbose=verbose)

    def get_metadata_schema(self) -> dict:
        """Return the metadata schema, allowing additional properties for Subject.

        Returns
        -------
        dict
            Metadata schema with additionalProperties enabled for the Subject block.
        """
        metadata_schema = super().get_metadata_schema()
        metadata_schema["additionalProperties"] = True
        metadata_schema["properties"]["Subject"]["additionalProperties"] = True

        return metadata_schema

    def get_metadata(self) -> dict:
        """Aggregate metadata from all interfaces and enrich with Alyx session data.

        Fetches session start time (with lab timezone), lab name, institution,
        task protocol, session description, and subject metadata from the Alyx REST API.

        Returns
        -------
        dict
            Metadata dictionary with NWBFile and Subject blocks populated.
        """
        metadata = super().get_metadata()  # Aggregates from the interfaces

        (session_metadata,) = self.one.alyx.rest(url="sessions", action="list", id=self.session)
        assert session_metadata["id"] == self.session, "Session metadata ID does not match the requested session ID."
        (lab_metadata,) = self.one.alyx.rest("labs", "list", name=session_metadata["lab"])

        session_start_time = datetime.fromisoformat(session_metadata["start_time"])
        tzinfo = tz.gettz(lab_metadata["timezone"])
        session_start_time = session_start_time.replace(tzinfo=tzinfo)
        metadata["NWBFile"]["session_start_time"] = session_start_time
        metadata["NWBFile"]["session_id"] = session_metadata["id"]
        metadata["NWBFile"]["lab"] = session_metadata["lab"].replace("lab", "").capitalize()
        metadata["NWBFile"]["institution"] = lab_metadata["institution"]
        if session_metadata.get("task_protocol"):
            task_protocol = session_metadata["task_protocol"]
            metadata["NWBFile"]["protocol"] = task_protocol
            session_description = f"The task protocol(s) performed in this experimental session:\n"
            # Determine protocol type and description from the mapping
            protocols = task_protocol.split("/")  # In case there are multiple protocols listed, separated by /
            for i, protocol in enumerate(protocols):
                protocol_type, protocol_description = get_protocol_type_and_description(protocol)
                if protocol_type is not None:
                    session_description = session_description + f"{i+1}. {protocol_description}\n"
            metadata["NWBFile"]["session_description"] = session_description
        # Setting publication and experiment description at project-specific converter level
        subject_metadata_block = get_ibl_subject_metadata(
            one=self.one, session_metadata=session_metadata, tzinfo=tzinfo
        )
        subject_metadata_block["weight"] = str(subject_metadata_block["weight"])  # Ensure weight is a string
        metadata["Subject"].update(subject_metadata_block)

        return metadata


class RawMesoscopeNWBConverter(IblConverter):
    """Primary conversion class for raw IBL mesoscope datasets."""

    def get_metadata(self) -> dict:
        """Return metadata merged with mesoscope general metadata YAML.

        Extends base metadata with experiment-level metadata loaded from
        `_metadata/mesoscope_general_metadata.yaml`.

        Returns
        -------
        dict
            Merged metadata dictionary.
        """
        metadata = super().get_metadata()

        mesoscope_metadata_file_path = Path(__file__).parent / "_metadata" / "mesoscope_general_metadata.yaml"
        experiment_metadata = load_dict_from_file(file_path=mesoscope_metadata_file_path)
        metadata = dict_deep_update(metadata, experiment_metadata)

        return metadata

    def temporally_align_data_interfaces(self, metadata: dict | None = None, conversion_options: dict | None = None):
        """Align raw imaging timestamps to the DAQ timeline.

        For every `RawImaging` interface, uses the `MesoscopeDAQInterface` to
        extract per-FOV timestamps from the Timeline DAQ recording and sets them
        as the aligned timestamps on the imaging interface.

        Parameters
        ----------
        metadata : dict | None, optional
            Metadata dictionary (not used; accepted for API compatibility).
        conversion_options : dict | None, optional
            Conversion options dictionary (not used; accepted for API compatibility).
        """
        if "DAQ" in self.data_interface_objects:
            daq_interface = self.data_interface_objects["DAQ"]
            for interface_name, interface in self.data_interface_objects.items():
                if "RawImaging" in interface_name:
                    FOV_name = interface.FOV_name
                    fov_timestamps = daq_interface.get_aligned_FOV_timestamps(FOV_name=FOV_name)
                    interface.set_aligned_timestamps(aligned_timestamps=fov_timestamps)


class ProcessedMesoscopeNWBConverter(IblConverter):
    """Primary conversion class for processed IBL mesoscope datasets."""

    def get_metadata(self) -> dict:
        """Return metadata merged with mesoscope general metadata YAML.

        Extends base metadata with experiment-level metadata loaded from
        `_metadata/mesoscope_general_metadata.yaml`.

        Returns
        -------
        dict
            Merged metadata dictionary.
        """
        metadata = super().get_metadata()

        mesoscope_metadata_file_path = Path(__file__).parent / "_metadata" / "mesoscope_general_metadata.yaml"
        experiment_metadata = load_dict_from_file(file_path=mesoscope_metadata_file_path)
        metadata = dict_deep_update(metadata, experiment_metadata)

        return metadata
