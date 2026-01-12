"""Data Interface for the pupil tracking."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools.nwb_helpers import get_module
from neuroconv.utils import load_dict_from_file
from pydantic import DirectoryPath
from pynwb import TimeSeries
from pynwb.behavior import PupilTracking


class PupilTrackingInterface(BaseDataInterface):
    """Interface for pupil tracking data."""

    def __init__(
        self,
        folder_path: DirectoryPath,
        camera_name: str,
    ):
        """Initialize the PupilTrackingInterface.
        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the folder containing the pupil tracking data files.
        camera_name : str
            Name of the camera (e.g., 'camera_left', 'camera_right', 'camera_body').

        """

        self.folder_path = Path(folder_path)

        super().__init__()
        self.camera_name = camera_name
        self.camera_view = re.search(r"(left|right|body)", camera_name).group(1)
        self.camera_times_file_name = f"_ibl_{camera_name}.times.npy"
        self.features_file_name = f"_ibl_{camera_name}.features.pqt"

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()

        pupils_metadata = load_dict_from_file(file_path=Path(__file__).parent.parent / "metadata" / "pupils.yml")
        metadata.update(pupils_metadata)

        return metadata

    def add_to_nwbfile(self, nwbfile, metadata: dict):
        camera_time = np.load(self.folder_path / self.camera_times_file_name, mmap_mode=None, allow_pickle=True)
        features = pd.read_parquet(self.folder_path / self.features_file_name)

        # Check for dimension mismatch between features and times
        features_len = len(features)
        times_len = len(camera_time)

        if features_len != times_len:
            import warnings

            if features_len > times_len:
                # Data is longer than timestamps - this is an error!
                # We have data samples without corresponding time information
                error_msg = (
                    f"Pupil tracking data for {self.camera_name} has "
                    f"more data samples ({features_len}) than timestamps ({times_len}). "
                    f"Cannot proceed without time information for all samples."
                )
                warnings.warn(error_msg, RuntimeWarning, stacklevel=2)
                raise RuntimeError(error_msg)
            else:
                # Timestamps are longer than data - we can truncate timestamps
                # This means we have extra timestamps at the end without corresponding data
                missing_samples = times_len - features_len
                warnings.warn(
                    f"Truncating timestamps for {self.camera_name}: "
                    f"timestamps length ({times_len}) exceeds features length ({features_len}) by {missing_samples} samples. "
                    f"Using first {features_len} timestamps.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                camera_time = camera_time[:features_len]

        pupil_time_series = list()
        for ibl_key in ["pupilDiameter_raw", "pupilDiameter_smooth"]:
            if ibl_key not in features:
                raise RuntimeError(f"Pupil tracking data for camera '{self.camera_name}' is missing column '{ibl_key}'")
            pupil_time_series.append(
                TimeSeries(
                    name=self.camera_view.capitalize() + metadata["Pupils"][ibl_key]["name"],
                    description=metadata["Pupils"][ibl_key]["description"],
                    data=np.array(features[ibl_key]),
                    timestamps=camera_time,
                    unit="px",
                )
            )
        # Normally best practice convention would be PupilTrackingLeft or PupilTrackingRight but
        # in this case I'd say LeftPupilTracking and RightPupilTracking reads better
        pupil_tracking = PupilTracking(
            name=f"{self.camera_view.capitalize()}PupilTracking", time_series=pupil_time_series
        )

        camera_module = get_module(nwbfile=nwbfile, name="camera", description="Processed camera data.")
        camera_module.add(pupil_tracking)
