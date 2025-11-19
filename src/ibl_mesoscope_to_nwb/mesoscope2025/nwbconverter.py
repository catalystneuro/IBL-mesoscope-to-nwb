"""Primary NWBConverter class for this dataset."""

from neuroconv import NWBConverter

from .datainterfaces import (
    IBLMesoscopeMotionCorrectedImagingInterface,
    IBLMesoscopeSegmentationInterface,
)


class ProcessedMesoscopeNWBConverter(NWBConverter):
    """Primary conversion class for processed IBL mesoscope datasets."""

    def __init__(self, source_data: dict, verbose: bool = True):
        """
        Initialize the ProcessedMesoscopeNWBConverter.

        Parameters
        ----------
        source_data : dict
            Dictionary mapping data interface names to their source data configurations.
            Keys should contain either 'Segmentation' or 'MotionCorrected' to be
            automatically mapped to the appropriate data interface class.
        verbose : bool, optional
            Whether to print verbose output during conversion, by default True.
        """
        data_interface_name_mapping = {
            "Segmentation": IBLMesoscopeSegmentationInterface,
            "MotionCorrectedImaging": IBLMesoscopeMotionCorrectedImagingInterface,
        }

        for interface_name in source_data.keys():
            if "Segmentation" in interface_name:
                for key, interface_class in data_interface_name_mapping.items():
                    if key in interface_name:
                        self.data_interface_classes[interface_name] = interface_class
            if "MotionCorrected" in interface_name:
                for key, interface_class in data_interface_name_mapping.items():
                    if key in interface_name:
                        self.data_interface_classes[interface_name] = interface_class

        super().__init__(source_data=source_data, verbose=verbose)
