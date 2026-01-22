from pathlib import Path

from ibl_to_nwb.converters._iblconverter import IblConverter
from neuroconv.utils import dict_deep_update, load_dict_from_file

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


class ProcessedMesoscopeNWBConverter(IblConverter):
    """Primary conversion class for processed IBL mesoscope datasets."""

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()

        mesoscope_metadata_file_path = Path(__file__).parent.parent / "_metadata" / "mesoscope_general_metadata.yml"
        experiment_metadata = load_dict_from_file(file_path=mesoscope_metadata_file_path)
        metadata = dict_deep_update(metadata, experiment_metadata)

        return metadata
