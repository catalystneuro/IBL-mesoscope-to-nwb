from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pydantic import DirectoryPath
from roiextractors import ImagingExtractor


class MesoscopeMotionCorrectedImagingExtractor(ImagingExtractor):
    """A segmentation extractor for IBL Motion Corrected Mesoscopic imaging data (.bin)."""

    extractor_name = "MesoscopeMotionCorrectedImagingExtractor"
    REVISION: str | None = None

    def __init__(self, one: ONE, session: str, FOV_name: str):
        self.one = one
        self.session = session
        self.revision = self.REVISION
        FOV_index = int(FOV_name.replace("FOV_", ""))
        self.plane_name = f"plane{FOV_index}"
        super().__init__()

        self._channel_names = ["OpticalChannel"]  # TODO update for dual plane

        self._timestamps = self.one.load_dataset(id=self.session, dataset="mpci.times", collection=f"alf/{FOV_name}")
        self._num_samples = len(self._timestamps)

        image_mean = self.one.load_dataset(
            id=self.session, dataset="mpciMeanImage.images", collection=f"alf/{FOV_name}"
        )
        self._frame_shape = (image_mean.shape[0], image_mean.shape[1])

        # Data type for binary file (Suite2p uses int16)
        self._dtype = np.dtype("int16")

        # Verify binary file size matches expected dimensions
        self._file_path = self.one.load_dataset(
            self.session, dataset="imaging.frames_motionRegistered", collection=f"suite2p/{self.plane_name}"
        )
        # TODO add correction fro "suite2"
        expected_size = self._num_samples * self._frame_shape[0] * self._frame_shape[1] * self._dtype.itemsize
        actual_size = self._file_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Binary file size mismatch. Expected {expected_size} bytes, got {actual_size} bytes. "
                f"Check dimensions: frames={self._num_samples}, height={self._frame_shape[0]}, "
                f"width={self._frame_shape[1]}, dtype={self._dtype}"
            )

    def get_sampling_frequency(self) -> None:
        """Get the sampling frequency in Hz.

        Returns
        -------
        sampling_frequency: float
            Sampling frequency in Hz.
        """
        return None

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._frame_shape

    def get_num_samples(self) -> int:
        """Get the number of samples in the video.

        Returns
        -------
        num_samples: int
            Number of samples in the video.
        """
        return self._num_samples

    def get_dtype(self) -> np.dtype:
        """Get the data type of the video frames.

        Returns
        -------
        dtype: numpy.dtype
            Data type of the video frames.
        """
        return self._dtype

    def get_channel_names(self) -> list:
        """Get the channel names in the recoding.

        Returns
        -------
        channel_names: list
            List of strings of channel names
        """
        return self._channel_names

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        """Get the series of samples.

        Parameters
        ----------
        start_sample: int, optional
            Start sample index (inclusive).
        end_sample: int, optional
            End sample index (exclusive).

        Returns
        -------
        series: numpy.ndarray
            The series of samples with shape (num_samples, height, width).

        Notes
        -----
        Importantly, we follow the convention that the dimensions of the array are returned in their matrix order,
        More specifically:
        (time, height, width)

        Which is equivalent to:
        (samples, rows, columns)

        For volumetric data, the dimensions are:
        (time, height, width, planes)

        Which is equivalent to:
        (samples, rows, columns, planes)

        Note that this does not match the cartesian convention:
        (t, x, y)

        Where x is the columns width or and y is the rows or height.
        """
        # Handle default values
        start_sample = 0 if start_sample is None else start_sample
        end_sample = self._num_samples if end_sample is None else end_sample

        # Validate indices
        if start_sample < 0 or start_sample >= self._num_samples:
            raise ValueError(f"start_sample {start_sample} out of range [0, {self._num_samples})")
        if end_sample <= start_sample or end_sample > self._num_samples:
            raise ValueError(f"end_sample {end_sample} out of range ({start_sample}, {self._num_samples}]")

        num_frames_to_read = end_sample - start_sample
        height, width = self._frame_shape

        # Memory-map the binary file for efficient reading
        # Suite2p saves data as (num_frames, height, width) in C order (row-major)
        data = np.memmap(self._file_path, dtype=self._dtype, mode="r", shape=(self._num_samples, height, width))

        # Extract the requested slice
        series = np.array(data[start_sample:end_sample])

        return series

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Retrieve the original unaltered timestamps for the data in this interface.

        This function should retrieve the data on-demand by re-initializing the IO.
        Can be overridden to return None if the extractor does not have native timestamps.

        Parameters
        ----------
        start_sample : int, optional
            The starting sample index. If None, starts from the beginning.
        end_sample : int, optional
            The ending sample index. If None, goes to the end.

        Returns
        -------
        timestamps: numpy.ndarray or None
            The timestamps for the data stream, or None if native timestamps are not available.
        """
        # Handle default values
        start_sample = 0 if start_sample is None else start_sample
        end_sample = self._num_samples if end_sample is None else end_sample

        # Validate indices
        if start_sample < 0 or start_sample >= self._num_samples:
            raise ValueError(f"start_sample {start_sample} out of range [0, {self._num_samples})")
        if end_sample <= start_sample or end_sample > self._num_samples:
            raise ValueError(f"end_sample {end_sample} out of range ({start_sample}, {self._num_samples}]")

        return self._timestamps[start_sample:end_sample]
