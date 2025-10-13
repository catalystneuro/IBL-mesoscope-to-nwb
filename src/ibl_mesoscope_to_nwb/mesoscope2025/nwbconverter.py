"""Primary NWBConverter class for this dataset."""

from neuroconv import NWBConverter
from .datainterfaces import IBLMesoscopeSegmentationInterface


class Mesoscope2025NWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Segmentation=IBLMesoscopeSegmentationInterface,
    )
