"""Data Interface for the special data type of ROI Motion Energy."""

import re
from pathlib import Path

import numpy as np
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools.nwb_helpers import get_module
from pydantic import DirectoryPath
from pynwb import TimeSeries


class RoiMotionEnergyInterface(BaseDataInterface):
    """Interface for ROI motion energy data."""

    def __init__(
        self,
        folder_path: DirectoryPath,
        camera_name: str,
    ):
        """Initialize the RoiMotionEnergyInterface.
        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the folder containing the ROI motion energy data files.
        camera_name : str
            Name of the camera (e.g., 'camera_left', 'camera_right', 'camera_body').

        """

        self.folder_path = Path(folder_path)

        super().__init__()
        self.camera_name = camera_name
        self.camera_view = re.search(r"(left|right|body)", camera_name).group(1)
        self.RME_file_name = f"{camera_name}.ROIMotionEnergy.npy"
        self.RME_times_file_name = f"_ibl_{camera_name}.times.npy"
        self.RME_position_file_name = f"{self.camera_view}ROIMotionEnergy.position.npy"

    def add_to_nwbfile(self, nwbfile, metadata: dict):
        camera_time = np.load(self.folder_path / self.RME_times_file_name, mmap_mode=None, allow_pickle=True)
        motion_energy_video_region = np.load(self.folder_path / self.RME_file_name, mmap_mode=None, allow_pickle=True)
        width, height, x, y = np.load(self.folder_path / self.RME_position_file_name, mmap_mode=None, allow_pickle=True)

        if camera_time.size == 0:
            raise RuntimeError(f"ROI motion energy timestamps for camera '{self.camera_name}' are empty")

        description = (
            f"Motion energy calculated for a region of the {self.camera_view} camera video that is {width} pixels "
            f"wide, {height} pixels tall, and the top-left corner of the region is the pixel ({x}, {y}).\n\n"
            "CAUTION: As each software will load the video in a different orientation, the ROI might need to be "
            "adapted. For example, when loading the video with cv2 in Python, x and y axes are flipped from the "
            f"convention used above. The region then becomes [{y}:{y + height}, {x}:{x + width}]."
        )

        motion_energy_series = TimeSeries(
            name=f"{self.camera_view.capitalize()}CameraMotionEnergy",
            description=description,
            data=motion_energy_video_region,
            timestamps=camera_time,
            unit="a.u.",
        )

        camera_module = get_module(nwbfile=nwbfile, name="camera", description="Processed camera data.")
        camera_module.add(motion_energy_series)
