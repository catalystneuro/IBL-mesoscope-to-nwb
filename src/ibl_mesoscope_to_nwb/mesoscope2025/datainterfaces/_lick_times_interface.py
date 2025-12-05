from pathlib import Path

import numpy as np
from hdmf.common import VectorData
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools.nwb_helpers import get_module
from pydantic import DirectoryPath
from pynwb import NWBFile
from pynwb.file import DynamicTable


class LickInterface(BaseDataInterface):
    """Interface for lick detection data."""

    def __init__(
        self,
        folder_path: DirectoryPath,
    ):
        """Initialize the LickInterface.
        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the folder containing the lick times file.

        """

        self.folder_path = Path(folder_path)

        super().__init__()

        # Lick times file name
        self._licks_times_file_name = "licks.times.npy"

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        """Add lick times to the NWB file.
        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to add the lick times to.
        metadata : dict
        """
        if not (self.folder_path / self._licks_times_file_name).is_file():
            print(
                f"Lick times file {self._licks_times_file_name} not found in folder {self.folder_path}. "
                "Lick times will not be added to the NWB file."
            )
            return
        licks = np.load(self.folder_path / self._licks_times_file_name, mmap_mode=None, allow_pickle=True)

        lick_events_table = DynamicTable(
            name="LickTimes",
            description=(
                "Time stamps of licks as detected from tongue dlc traces. "
                "If left and right camera exist, the licks detected from both cameras are combined."
            ),
            columns=[
                VectorData(
                    name="lick_time",
                    description="Time stamps of licks as detected from tongue dlc traces",
                    data=licks,
                )
            ],
        )

        camera_module = get_module(nwbfile=nwbfile, name="camera", description="Processed camera data.")
        camera_module.add(lick_events_table)
