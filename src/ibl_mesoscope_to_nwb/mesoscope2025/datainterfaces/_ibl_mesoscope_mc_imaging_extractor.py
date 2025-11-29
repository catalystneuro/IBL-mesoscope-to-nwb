from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pydantic import DirectoryPath
from roiextractors import ImagingExtractor


class IBLMesoscopeMotionCorrectedImagingExtractor(ImagingExtractor):
    """A segmentation extractor for IBL Motion Corrected Mesoscopic imaging data (.bin)."""

    extractor_name = "IBLMesoscopeMotionCorrectedImagingExtractor"

    @classmethod
    def get_available_planes(cls, folder_path: DirectoryPath) -> list[str]:
        """Get the available plane names from the folder produced by IBL Mesoscope.

        Parameters
        ----------
        folder_path : PathType
            Path to IBL Mesoscope output path.

        Returns
        -------
        FOV_names: list
            List of plane names.
        """
        from natsort import natsorted

        folder_path = Path(folder_path)
        prefix = "plane"
        plane_paths = natsorted(folder_path.glob(pattern=prefix + "*"))
        assert len(plane_paths), f"No planes found in '{folder_path}'."
        FOV_names = [plane_path.stem for plane_path in plane_paths]
        return FOV_names

    def __init__(self, file_path: str):
        """Initialize a IBLMesoscopeMotionCorrectedImagingExtractor instance.

        Main class for extracting imaging data from .bin format.

        Parameters
        ----------
        file_path: str or Path
            Path to the .bin file containing imaging data.
            Expected path format: .../suite2p/planeX/imaging.frames_motionRegistered.bin
        """
        super().__init__(file_path=file_path)
        self._file_path = Path(file_path)
        self._channel_names = ["green_channel"]

        # Validate file path structure
        if not self._file_path.name == "imaging.frames_motionRegistered.bin":
            raise ValueError(f"Expected file named 'imaging.frames_motionRegistered.bin', got '{self._file_path.name}'")

        # Extract plane number from path (e.g., plane0 -> 0)
        plane_folder = self._file_path.parent.name
        if not plane_folder.startswith("plane"):
            raise ValueError(f"Expected parent folder to start with 'plane', got '{plane_folder}'")

        plane_number = int(plane_folder.replace("plane", ""))

        # Find session folder (2 levels up from suite2p/planeX/)
        session_folder = self._file_path.parent.parent.parent

        # Construct path to corresponding FOV folder
        FOV_name = f"FOV_{plane_number:02d}"
        self._alf_folder = session_folder / "alf" / FOV_name

        if not self._alf_folder.exists():
            raise FileNotFoundError(f"ALF folder not found: {self._alf_folder}")

        # Load timestamps to determine number of frames
        timestamps_file = self._alf_folder / "mpci.times.npy"
        if not timestamps_file.exists():
            raise FileNotFoundError(f"Timestamps file not found: {timestamps_file}")

        self._timestamps = np.load(timestamps_file)
        self._num_samples = len(self._timestamps)

        # Load mean image to get frame dimensions
        mean_image_file = self._alf_folder / "mpciMeanImage.images.npy"
        if not mean_image_file.exists():
            raise FileNotFoundError(f"Mean image file not found: {mean_image_file}")

        mean_image = np.load(mean_image_file)
        self._image_shape = mean_image.shape  # Should be (height, width)

        # Data type for binary file (Suite2p uses int16)
        self._dtype = np.dtype("int16")

        # Verify binary file size matches expected dimensions
        expected_size = self._num_samples * self._image_shape[0] * self._image_shape[1] * self._dtype.itemsize
        actual_size = self._file_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Binary file size mismatch. Expected {expected_size} bytes, got {actual_size} bytes. "
                f"Check dimensions: frames={self._num_samples}, height={self._image_shape[0]}, "
                f"width={self._image_shape[1]}, dtype={self._dtype}"
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
        return self._image_shape

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
        height, width = self._image_shape

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
