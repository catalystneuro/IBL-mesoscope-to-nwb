"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from neuroconv.datainterfaces import (
    SpikeGLXRecordingInterface,
    PhySortingInterface,
)

from ibl_mesoscope_to_nwb.mesoscope2025 import Mesoscope2025BehaviorInterface


class Mesoscope2025NWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Recording=SpikeGLXRecordingInterface,
        Sorting=PhySortingInterface,
        Behavior=Mesoscope2025BehaviorInterface,
    )
