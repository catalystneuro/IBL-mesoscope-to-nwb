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
        from .datainterfaces import IBLMesoscopeRawImagingInterface, RawVideoInterface

        for interface_name in source_data.keys():
            if "RawImaging" in interface_name:
                self.data_interface_classes[interface_name] = IBLMesoscopeRawImagingInterface
            if "RawVideo" in interface_name:
                self.data_interface_classes[interface_name] = RawVideoInterface

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
            BrainwideMapTrialsInterface,
            IBLMesoscopeAnatomicalLocalizationInterface,
            IBLMesoscopeMotionCorrectedImagingInterface,
            IBLMesoscopeSegmentationInterface,
            LickInterface,
            PupilTrackingInterface,
            RoiMotionEnergyInterface,
            WheelInterface,
        )

        for interface_name in source_data.keys():
            if "AnatomicalLocalization" in interface_name:
                self.data_interface_classes[interface_name] = IBLMesoscopeAnatomicalLocalizationInterface
            if "Segmentation" in interface_name:
                self.data_interface_classes[interface_name] = IBLMesoscopeSegmentationInterface
            if "MotionCorrected" in interface_name:
                self.data_interface_classes[interface_name] = IBLMesoscopeMotionCorrectedImagingInterface
            if "Lick" in interface_name:
                self.data_interface_classes[interface_name] = LickInterface
            if "Wheel" in interface_name:
                self.data_interface_classes[interface_name] = WheelInterface
            if "ROIMotionEnergy" in interface_name:
                self.data_interface_classes[interface_name] = RoiMotionEnergyInterface
            if "PupilTracking" in interface_name:
                self.data_interface_classes[interface_name] = PupilTrackingInterface
            if "Trials" in interface_name:
                self.data_interface_classes[interface_name] = BrainwideMapTrialsInterface

        super().__init__(source_data=source_data, verbose=verbose)
