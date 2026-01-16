"""Primary NWBConverter class for this dataset."""

from datetime import datetime

from neuroconv import BaseDataInterface, ConverterPipe
from one.api import ONE


# class RawMesoscopeNWBConverter(NWBConverter):
#     """Primary conversion class for my extracellular electrophysiology dataset."""

#     def __init__(self, source_data: dict, verbose: bool = True):
#         """
#         Initialize the RawMesoscopeNWBConverter.

#         Parameters
#         ----------
#         source_data : dict
#             Dictionary of source data for each data interface.
#         verbose : bool, optional
#             If True, print verbose output, by default True.
#         """
#         from .datainterfaces import IBLMesoscopeRawImagingInterface

#         data_interface_name_mapping = {
#             "RawImaging": IBLMesoscopeRawImagingInterface,
#         }

#         for interface_name in source_data.keys():
#             if "RawImaging" in interface_name:
#                 for key, interface_class in data_interface_name_mapping.items():
#                     if key in interface_name:
#                         self.data_interface_classes[interface_name] = interface_class

#         super().__init__(source_data=source_data, verbose=verbose)


class ProcessedMesoscopeNWBConverter(ConverterPipe):
    """Primary conversion class for processed IBL mesoscope datasets."""

    def __init__(
        self,
        one: ONE,
        eid: str,
        data_interfaces: list[BaseDataInterface] | dict[str, BaseDataInterface],
        verbose=False,
    ):
        self.one = one
        self.eid = eid
        super().__init__(data_interfaces=data_interfaces, verbose=verbose)

    def get_metadata(self):
        metadata = super().get_metadata()

        try:
            ((session_metadata),) = self.one.alyx.rest(url="sessions", action="list", id=self.eid)
        except Exception as e:
            raise RuntimeError(f"Failed to access ONE for eid {self.eid}: {e}")

        session_start_time = datetime.fromisoformat(session_metadata["start_time"])
        metadata["NWBFile"]["session_start_time"] = session_start_time
        metadata["NWBFile"]["session_id"] = self.eid
        metadata["Subject"]["subject_id"] = session_metadata["subject"]

        return metadata
