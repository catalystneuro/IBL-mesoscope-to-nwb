"""Primary NWBConverter class for this dataset."""

from neuroconv import NWBConverter

from .datainterfaces import IBLMesoscopeRawImagingInterface


class RawMesoscope2025NWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    def __init__(self, source_data: dict, verbose: bool = True):
        data_interface_name_mapping = {
            "RawImaging": IBLMesoscopeRawImagingInterface,
        }

        for interface_name in source_data.keys():
            if "RawImaging" in interface_name:
                for key, interface_class in data_interface_name_mapping.items():
                    if key in interface_name:
                        self.data_interface_classes[interface_name] = interface_class

        super().__init__(source_data=source_data, verbose=verbose)
