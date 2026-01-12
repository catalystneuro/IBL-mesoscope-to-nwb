from pathlib import Path

import numpy as np
from brainbox.behavior import wheel as wheel_methods
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools.nwb_helpers import get_module
from neuroconv.utils import load_dict_from_file
from pydantic import DirectoryPath
from pynwb import TimeSeries
from pynwb.behavior import CompassDirection, SpatialSeries
from pynwb.epoch import TimeIntervals


class WheelInterface(BaseDataInterface):
    """Interface for wheel movement data."""

    def __init__(
        self,
        folder_path: DirectoryPath,
    ):
        """Initialize the WheelInterface.
        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the folder containing the wheel movement data files.

        """

        self.folder_path = Path(folder_path)

        super().__init__()
        self._wheel_position_file_name = "_ibl_wheel.position.npy"
        self._wheel_timestamps_file_name = "_ibl_wheel.timestamps.npy"
        self._wheel_moves_intervals_file_name = "_ibl_wheelMoves.intervals.npy"
        self._wheel_moves_peak_amplitude_file_name = "_ibl_wheelMoves.peakAmplitude.npy"

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()

        metadata.update(load_dict_from_file(file_path=Path(__file__).parent.parent / "metadata" / "wheel.yml"))

        return metadata

    def add_to_nwbfile(self, nwbfile, metadata: dict, stub_test: bool = False, stub_duration: float = 10.0):
        """
        Add wheel movement data to NWBFile.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWBFile to add data to.
        metadata : dict
            Metadata dictionary.
        stub_test : bool, default: False
            If True, only add the first stub_duration seconds of data for testing.
        stub_duration : float, default: 10.0
            Duration in seconds to include when stub_test=True.
        """
        wheel_position = np.load(self.folder_path / self._wheel_position_file_name, mmap_mode=None, allow_pickle=True)
        wheel_timestamps = np.load(
            self.folder_path / self._wheel_timestamps_file_name, mmap_mode=None, allow_pickle=True
        )
        wheel_moves_intervals = np.load(
            self.folder_path / self._wheel_moves_intervals_file_name, mmap_mode=None, allow_pickle=True
        )
        wheel_moves_peak_amplitude = np.load(
            self.folder_path / self._wheel_moves_peak_amplitude_file_name, mmap_mode=None, allow_pickle=True
        )

        # Subset data if stub_test
        if stub_test:
            original_times = wheel_timestamps.copy()
            original_position = wheel_position.copy()

            if original_times.size == 0:
                raise ValueError("Wheel timestamps array is empty; cannot create stub dataset.")

            stub_limit = original_times[0] + stub_duration
            time_mask = original_times <= stub_limit
            if not time_mask.any():
                sample_limit = min(1000, original_times.size)
                time_mask = np.zeros_like(original_times, dtype=bool)
                time_mask[:sample_limit] = True

            wheel_timestamps = original_times[time_mask]
            wheel_position = original_position[time_mask]

            interval_mask = wheel_moves_intervals[:, 0] <= stub_limit
            if not interval_mask.any():
                interval_mask = np.zeros(len(wheel_moves_intervals), dtype=bool)
                interval_mask[: min(100, len(interval_mask))] = True
            wheel_moves_intervals = wheel_moves_intervals[interval_mask]
            wheel_moves_peak_amplitude = wheel_moves_peak_amplitude[interval_mask]

        if wheel_timestamps.size < 2:
            raise ValueError("Wheel timestamps must contain at least two samples.")

        # Estimate velocity and acceleration
        interpolation_frequency = 1000.0  # Hz
        interpolated_position, interpolated_timestamps = wheel_methods.interpolate_position(
            re_ts=wheel_timestamps, re_pos=wheel_position, freq=interpolation_frequency
        )
        velocity, acceleration = wheel_methods.velocity_filtered(pos=interpolated_position, fs=interpolation_frequency)

        # Deterministically regular
        interpolated_starting_time = interpolated_timestamps[0]
        interpolated_rate = 1 / (interpolated_timestamps[1] - interpolated_timestamps[0])

        # Wheel intervals of movement
        wheel_movement_intervals = TimeIntervals(
            name="WheelMovementIntervals",
            description=metadata["WheelMovement"]["description"],
        )
        for start_time, stop_time in wheel_moves_intervals:
            wheel_movement_intervals.add_row(start_time=start_time, stop_time=stop_time)
        wheel_movement_intervals.add_column(
            name=metadata["WheelMovement"]["columns"]["peakAmplitude"]["name"],
            description=metadata["WheelMovement"]["columns"]["peakAmplitude"]["description"],
            data=wheel_moves_peak_amplitude,
        )

        # Wheel position over time
        compass_direction = CompassDirection(
            spatial_series=SpatialSeries(
                name=metadata["WheelPosition"]["name"],
                description=metadata["WheelPosition"]["description"],
                data=wheel_position,
                timestamps=wheel_timestamps,
                unit="radians",
                reference_frame="Initial angle at start time is zero. Counter-clockwise is positive.",
            )
        )
        velocity_series = TimeSeries(
            name=metadata["WheelVelocity"]["name"],
            description=metadata["WheelVelocity"]["description"],
            data=velocity,
            starting_time=interpolated_starting_time,
            rate=interpolated_rate,
            unit="rad/s",
        )
        acceleration_series = TimeSeries(
            name=metadata["WheelAcceleration"]["name"],
            description=metadata["WheelAcceleration"]["description"],
            data=acceleration,
            starting_time=interpolated_starting_time,
            rate=interpolated_rate,
            unit="rad/s^2",
        )

        behavior_module = get_module(nwbfile=nwbfile, name="wheel", description="Processed wheel data.")
        behavior_module.add(wheel_movement_intervals)
        behavior_module.add(compass_direction)
        behavior_module.add(velocity_series)
        behavior_module.add(acceleration_series)
