"""A segmentation extractor for Suite2p.

Classes
-------
IBLMesoscopeSegmentationExtractor
    A segmentation extractor for Suite2p.
"""

from pathlib import Path
from typing import List
from warnings import warn

import numpy as np
import pandas as pd
from pydantic import DirectoryPath
from roiextractors import SegmentationExtractor
from roiextractors.extraction_tools import _image_mask_extractor
from roiextractors.segmentationextractor import _RoiResponse


class IBLMesoscopeSegmentationExtractor(SegmentationExtractor):
    """A segmentation extractor for IBL Mesoscope."""

    extractor_name = "IBLMesoscopeSegmentationExtractor"

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
        prefix = "FOV_"
        fov_paths = natsorted(folder_path.glob(pattern=prefix + "*"))
        assert len(fov_paths), f"No planes found in '{folder_path}'."
        FOV_names = [fov_path.stem for fov_path in fov_paths]
        return FOV_names

    def __init__(
        self,
        folder_path: DirectoryPath,
        FOV_name: str | None = None,
    ):
        """Create SegmentationExtractor object out of suite 2p data type.

        Parameters
        ----------
        folder_path: str or Path
            The path to the 'alf' folder, where processed imaging data is stored.
        FOV_name: str, optional
            The name of the plane to load, to determine what planes are available use IBLMesoscopeSegmentationExtractor.get_available_planes(folder_path).
        """

        FOV_names = self.get_available_planes(folder_path=folder_path)
        if FOV_name is None:
            if len(FOV_names) > 1:
                # For backward compatibility maybe it is better to warn first
                warn(
                    "More than one plane is detected! Please specify which plane you wish to load with the `FOV_name` argument. "
                    "To see what planes are available, call `IBLMesoscopeSegmentationExtractor.get_available_planes(folder_path=...)`.",
                    UserWarning,
                )
            FOV_name = FOV_names[0]

        if FOV_name not in FOV_names:
            raise ValueError(
                f"The selected plane '{FOV_name}' is not a valid plane name. To see what planes are available, "
                f"call `IBLMesoscopeSegmentationExtractor.get_available_planes(folder_path=...)`."
            )
        self.FOV_name = FOV_name

        super().__init__()

        self.folder_path = Path(folder_path)

        self._timestamps_file_name = "mpci.times.npy"
        # ROIs temporal and spatial components
        self._raw_traces_file_name = "mpci.ROIActivityF.npy"
        self._deconvolved_traces_file_name = "mpci.ROIActivityDeconvolved.npy"
        self._ROIs_iscell_file_name = "mpciROIs.mpciROITypes.npy"
        self._ROI_masks_file_name = "mpciROIs.masks.sparse_npz"
        self._ROIs_classifier_file_name = "mpciROIs.cellClassifier.npy"
        self._ROIs_location_file_name = "mpciROIs.stackPos.npy"
        # neuropil temporal and spatial components
        self._neuropil_traces_file_name = "mpci.ROIActivityNeuropilF.npy"
        self._neuropil_masks_file_name = "mpciROIs.neuropilMasks.sparse_npz"
        # summary images
        self._mean_image_file_name = "mpciMeanImage.images.npy"
        # ROI uuids
        self._ROIs_uuids_file_name = "mpciROIs.uuids.csv"

        self._channel_names = ["OpticalChannel"]

        self.set_property(key="Classifier", values=self.cell_classifier, ids=range(len(self.cell_classifier)))
        self.set_property(key="UUID", values=self.uuids, ids=range(len(self.uuids)))

    def _load_npy(self, file_name: str, mmap_mode=None, transpose: bool = False, require: bool = False):
        """Load a .npy file with specified filename. Returns None if file is missing.

        Parameters
        ----------
        file_name: str
            The name of the .npy file to load.
        mmap_mode: str
            The mode to use for memory mapping. See numpy.load for details.
        transpose: bool, optional
            Whether to transpose the loaded array.
        require: bool, optional
            Whether to raise an error if the file is missing.

        Returns
        -------
            The loaded .npy file.
        """
        file_path = self.folder_path / self.FOV_name / file_name
        if not file_path.exists():
            if require:
                raise FileNotFoundError(f"File {file_path} not found.")
            return

        data = np.load(file_path, mmap_mode=mmap_mode, allow_pickle=mmap_mode is None)
        if transpose:
            return data.T

        return data

    def get_accepted_list(self) -> list[int]:
        if not hasattr(self, "iscell"):
            self.iscell = self._load_npy(file_name=self._ROIs_iscell_file_name, require=True)
            assert self.iscell is not None, f"{self._ROIs_iscell_file_name} is required but could not be loaded"
        return list(np.where(self.iscell == 1)[0])

    def get_rejected_list(self) -> list[int]:
        if not hasattr(self, "iscell"):
            self.iscell = self._load_npy(file_name=self._ROIs_iscell_file_name, require=True)
            assert self.iscell is not None, f"{self._ROIs_iscell_file_name} is required but could not be loaded"
        return list(np.where(self.iscell == 0)[0])

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        timestamps = self._load_npy(file_name=self._timestamps_file_name, require=False)
        if timestamps is None:
            return None
        return timestamps[start_sample:end_sample]

    def get_frame_shape(self) -> tuple[int, int]:
        """Get the shape of the frames in the recording.

        Returns
        -------
        frame_shape: tuple[int, int]
            Shape of the frames in the recording.
        """
        if not hasattr(self, "_frame_shape"):
            image_mean = self._load_npy(file_name=self._mean_image_file_name, require=True)
            assert image_mean is not None, f"{self._mean_image_file_name} is required but could not be loaded"
            self._frame_shape = (image_mean.shape[0], image_mean.shape[1])
        return self._frame_shape

    def get_num_samples(self) -> int:
        """Get the number of samples in the recording (duration of recording).

        Returns
        -------
        num_samples: int
            Number of samples in the recording.
        """
        if not hasattr(self, "_num_frames"):
            times = self._load_npy(file_name=self._timestamps_file_name, require=True)
            assert times is not None, f"{self._timestamps_file_name} is required but could not be loaded"
            self._num_frames = times.shape[0]
        return self._num_frames

    @property
    def cell_classifier(self) -> np.ndarray:
        """Returns the cell classifier values for each ROI."""
        cell_classifier = self._load_npy(file_name=self._ROIs_classifier_file_name, require=True)
        assert cell_classifier is not None, f"{self._ROIs_classifier_file_name} is required but could not be loaded"
        return cell_classifier

    @property
    def uuids(self) -> list[str]:
        """Returns the UUIDs for each ROI."""
        csv_file_path = self.folder_path / self.FOV_name / self._ROIs_uuids_file_name
        df = pd.read_csv(csv_file_path, usecols=["uuids"])
        uuids = df["uuids"].to_list()
        assert uuids is not None, f"{self._ROIs_uuids_file_name} is required but could not be loaded"
        return uuids

    def get_roi_locations(self, roi_ids=None) -> np.ndarray:
        """Returns the center locations (x, y, z) of each ROI."""
        if not hasattr(self, "roi_locations"):
            roi_locations = self._load_npy(file_name=self._ROIs_location_file_name, require=True, transpose=True)
            assert roi_locations is not None, f"{self._ROIs_location_file_name} is required but could not be loaded"
            self.roi_locations = roi_locations if roi_ids is None else roi_locations[roi_ids]
        return self.roi_locations

    def get_roi_pixel_masks(self, roi_ids=None) -> list[np.ndarray]:
        """Get the pixel masks for the specified ROIs in sparse format.

        Parameters
        ----------
        roi_ids : array-like, optional
            List of ROI IDs to get the pixel masks for. If None, all ROIs are returned.

        Returns
        -------
        pixel_masks : list[np.ndarray]
            List of arrays, where each array is of shape (n_pixels, 3) with columns [y, x, weight].
            Each row represents a pixel in the ROI mask with its y-coordinate, x-coordinate, and weight.

        Notes
        -----
        The IBL format uses pydata/sparse library (not scipy.sparse) to save masks with sparse.save_npz().
        The masks are stored as a 3D array with shape (num_rois, Ly, Lx).
        """
        import sparse

        # Load the sparse masks file (using pydata/sparse, not scipy.sparse)
        file_path = self.folder_path / self.FOV_name / self._ROI_masks_file_name
        with open(file_path, "rb") as fp:
            masks = sparse.load_npz(fp)  # shape: (num_rois, Ly, Lx)

        # Determine which ROIs to process
        if roi_ids is None:
            roi_indices = range(self.get_num_rois())
        else:
            roi_indices = roi_ids

        pixel_masks = []
        for roi_idx in roi_indices:
            # Get the 2D mask for this ROI
            roi_mask = masks[roi_idx]  # shape: (Ly, Lx)

            # Convert to COO format to get coordinates and weights
            roi_mask_coo = sparse.COO(roi_mask)

            # Extract y, x coordinates and weights
            y_coords = roi_mask_coo.coords[0]
            x_coords = roi_mask_coo.coords[1]
            weights = roi_mask_coo.data

            # Stack into (n_pixels, 3) array: [y, x, weight]
            pixel_mask = np.vstack([y_coords, x_coords, weights]).T
            pixel_masks.append(pixel_mask)

        return pixel_masks

    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        self._image_masks = _image_mask_extractor(
            self.get_roi_pixel_masks(),
            roi_ids if roi_ids is not None else list(range(self.get_num_rois())),
            self.get_frame_shape(),
        )
        return self._image_masks

    def get_background_pixel_masks(self, background_ids=None) -> list[np.ndarray]:
        """Get the pixel masks for the specified background (neuropil) ROIs in sparse format.

        Parameters
        ----------
        background_ids : array-like, optional
            List of background ROI IDs to get the pixel masks for. If None, all background ROIs are returned.

        Returns
        -------
        pixel_masks : list[np.ndarray]
            List of arrays, where each array is of shape (n_pixels, 3) with columns [y, x, weight].
            Each row represents a pixel in the background mask with its y-coordinate, x-coordinate, and weight.

        Notes
        -----
        The IBL format uses pydata/sparse library (not scipy.sparse) to save masks with sparse.save_npz().
        The neuropil masks are stored as a 3D array with shape (num_background_rois, Ly, Lx).
        """
        import sparse

        # Load the sparse neuropil masks file (using pydata/sparse, not scipy.sparse)
        file_path = self.folder_path / self.FOV_name / self._neuropil_masks_file_name
        with open(file_path, "rb") as fp:
            masks = sparse.load_npz(fp)  # shape: (num_background_rois, Ly, Lx)

        # Determine which background ROIs to process
        if background_ids is None:
            background_indices = range(self.get_num_background_components())
        else:
            background_indices = background_ids

        pixel_masks = []
        for bg_idx in background_indices:
            # Get the 2D mask for this background ROI
            bg_mask = masks[bg_idx]  # shape: (Ly, Lx)

            # Convert to COO format to get coordinates and weights
            bg_mask_coo = sparse.COO(bg_mask)

            # Extract y, x coordinates and weights
            y_coords = bg_mask_coo.coords[0]
            x_coords = bg_mask_coo.coords[1]
            weights = bg_mask_coo.data

            # Stack into (n_pixels, 3) array: [y, x, weight]
            pixel_mask = np.vstack([y_coords, x_coords, weights]).T
            pixel_masks.append(pixel_mask)

        return pixel_masks

    def get_background_image_masks(self, roi_ids=None) -> np.ndarray:
        self._background_image_masks = _image_mask_extractor(
            self.get_background_pixel_masks(),
            roi_ids if roi_ids is not None else list(range(self.get_num_background_components())),
            self.get_frame_shape(),
        )
        return self._background_image_masks

    def _get_rois_responses(self) -> List[_RoiResponse]:
        """Load the ROI responses from the Suite2p output files.
        Returns
        -------
        _roi_responses: List[_RoiResponse]
            List of _RoiResponse objects containing the ROI responses.
        """
        if not self._roi_responses:
            self._roi_responses = []

            raw_traces = self._load_npy(file_name=self._raw_traces_file_name, mmap_mode="r")
            neuropil_traces = self._load_npy(file_name=self._neuropil_traces_file_name, mmap_mode="r")
            deconvolved_traces = self._load_npy(file_name=self._deconvolved_traces_file_name, mmap_mode="r")

            cell_ids = None
            if raw_traces is not None:
                cell_ids = list(range(raw_traces.shape[1]))
                self._roi_responses.append(
                    _RoiResponse("raw", raw_traces, cell_ids)
                )  # TODO check if it is raw or DF over F

            if neuropil_traces is not None:
                if cell_ids is None:
                    cell_ids = list(range(neuropil_traces.shape[1]))
                self._roi_responses.append(_RoiResponse("neuropil", neuropil_traces, list(cell_ids)))

            if deconvolved_traces is not None:
                if cell_ids is None:
                    cell_ids = list(range(deconvolved_traces.shape[1]))
                self._roi_responses.append(_RoiResponse("deconvolved", deconvolved_traces, list(cell_ids)))

        return self._roi_responses

    def get_traces_dict(self) -> dict:
        """Get traces as a dictionary with key as the name of the ROiResponseSeries.

        Returns
        -------
        _roi_response_dict: dict
            dictionary with key, values representing different types of RoiResponseSeries:
                Raw Fluorescence, DeltaFOverF, Denoised, Neuropil, Deconvolved, Background, etc.
        """
        if not self._roi_responses:
            self._get_rois_responses()

        traces = {response.response_type: response.data for response in self._roi_responses}
        return traces

    def get_images_dict(self) -> dict:
        """Get images as a dictionary with key as the name of the ROIResponseSeries.

        Returns
        -------
        _summary_images: dict
            dictionary with key, values representing different types of Images used in segmentation:
                Mean, Correlation image, Maximum projection, etc.
        """
        if not self._summary_images:
            self._summary_images = {}
            mean_image = self._load_npy(file_name=self._mean_image_file_name, require=False)
            if mean_image is not None:
                self._summary_images["mean"] = mean_image
        return self._summary_images

    def has_time_vector(self) -> bool:
        """Detect if the SegmentationExtractor has a time vector set or not.

        Returns
        -------
        has_time_vector: bool
            True if the SegmentationExtractor has a time vector set, otherwise False.
        """
        self._times = self.get_timestamps()
        return self._times is not None
