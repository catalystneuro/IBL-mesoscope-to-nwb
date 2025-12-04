"""Primary NWBConverter class for this dataset."""

from neuroconv import NWBConverter


class RawMesoscopeNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    def __init__(self, source_data: dict, verbose: bool = True):
        """
        Initialize the RawMesoscopeNWBConverter.

        Parameters
        ----------
        source_data : dict
            Dictionary of source data for each data interface.
        verbose : bool, optional
            If True, print verbose output, by default True.
        """
        from .datainterfaces import IBLMesoscopeRawImagingInterface

        data_interface_name_mapping = {
            "RawImaging": IBLMesoscopeRawImagingInterface,
        }

        for interface_name in source_data.keys():
            if "RawImaging" in interface_name:
                for key, interface_class in data_interface_name_mapping.items():
                    if key in interface_name:
                        self.data_interface_classes[interface_name] = interface_class

        super().__init__(source_data=source_data, verbose=verbose)


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
        from .datainterfaces import (
            IBLMesoscopeMotionCorrectedImagingInterface,
            IBLMesoscopeSegmentationInterface,
            IBLMesoscopeAnatomicalLocalizationInterface,
            LickInterface,
        )

        data_interface_name_mapping = {
            "Segmentation": IBLMesoscopeSegmentationInterface,
            "MotionCorrectedImaging": IBLMesoscopeMotionCorrectedImagingInterface,
            "AnatomicalLocalization": IBLMesoscopeAnatomicalLocalizationInterface,
            "Lick": LickInterface,
        }

        for interface_name in source_data.keys():
            if "AnatomicalLocalization" in interface_name:
                for key, interface_class in data_interface_name_mapping.items():
                    if key in interface_name:
                        self.data_interface_classes[interface_name] = interface_class
            if "Segmentation" in interface_name:
                for key, interface_class in data_interface_name_mapping.items():
                    if key in interface_name:
                        self.data_interface_classes[interface_name] = interface_class
            if "MotionCorrected" in interface_name:
                for key, interface_class in data_interface_name_mapping.items():
                    if key in interface_name:
                        self.data_interface_classes[interface_name] = interface_class
            if "Lick" in interface_name:
                for key, interface_class in data_interface_name_mapping.items():
                    if key in interface_name:
                        self.data_interface_classes[interface_name] = interface_class

        super().__init__(source_data=source_data, verbose=verbose)
