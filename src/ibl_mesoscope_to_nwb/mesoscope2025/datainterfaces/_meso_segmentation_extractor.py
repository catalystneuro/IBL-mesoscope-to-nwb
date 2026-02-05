"""A segmentation extractor for Suite2p.

Classes
-------
MesoscopeSegmentationExtractor
    A segmentation extractor for Suite2p.
"""

from typing import List

import numpy as np
import sparse
from one.api import ONE
from roiextractors.extraction_tools import _image_mask_extractor
from roiextractors.segmentationextractor import (
    SegmentationExtractor,
    _ROIMasks,
    _RoiResponse,
)


class MesoscopeSegmentationExtractor(SegmentationExtractor):
    """A segmentation extractor for IBL Mesoscope."""

    extractor_name = "MesoscopeSegmentationExtractor"
    REVISION: str | None = None

    def __init__(self, one: ONE, session: str, FOV_name: str):
        self.one = one
        self.session = session
        self.revision = self.REVISION
        self.FOV_name = FOV_name
        self.add_background = False

        available_datasets = self.one.list_datasets(
            eid=self.session, filename="mpciROIs.neuropilMasks.masks.json", collection=f"alf/{self.FOV_name}"
        )
        if len(available_datasets) > 0:
            self.add_background = True
        super().__init__()

        self._channel_names = ["OpticalChannel"]  # TODO update for dual plane

        self.set_property(key="Classifier", values=self.cell_classifier, ids=range(len(self.cell_classifier)))
        self.set_property(key="UUID", values=self.uuids, ids=range(len(self.uuids)))

    def get_accepted_list(self) -> list[int]:
        if not hasattr(self, "iscell"):
            self.iscell = self.one.load_dataset(
                id=self.session, dataset="mpciROIs.mpciROITypes", collection=f"alf/{self.FOV_name}"
            )
        return list(np.where(self.iscell == 1)[0])

    def get_rejected_list(self) -> list[int]:
        if not hasattr(self, "iscell"):
            self.iscell = self.one.load_dataset(
                id=self.session, dataset="mpciROIs.mpciROITypes", collection=f"alf/{self.FOV_name}"
            )
        return list(np.where(self.iscell == 0)[0])

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        timestamps = self.one.load_dataset(id=self.session, dataset="mpci.times", collection=f"alf/{self.FOV_name}")
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
            image_mean = self.one.load_dataset(
                id=self.session, dataset="mpciMeanImage.images", collection=f"alf/{self.FOV_name}"
            )
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
            times = self.one.load_dataset(id=self.session, dataset="mpci.times", collection=f"alf/{self.FOV_name}")
            self._num_frames = times.shape[0]
        return self._num_frames

    @property
    def cell_classifier(self) -> np.ndarray:
        """Returns the cell classifier values for each ROI."""
        cell_classifier = self.one.load_dataset(
            id=self.session, dataset="mpciROIs.cellClassifier", collection=f"alf/{self.FOV_name}"
        )
        return cell_classifier

    @property
    def uuids(self) -> list[str]:
        """Returns the UUIDs for each ROI."""
        uuids = self.one.load_dataset(id=self.session, dataset="mpciROIs.uuids", collection=f"alf/{self.FOV_name}")
        return uuids

    def get_roi_locations(self, roi_ids=None) -> np.ndarray:
        """Returns the center locations (x, y, z) of each ROI."""
        if not hasattr(self, "roi_locations"):
            roi_locations = self.one.load_dataset(
                id=self.session, dataset="mpciROIs.stackPos", collection=f"alf/{self.FOV_name}"
            )
            self.roi_locations = roi_locations if roi_ids is None else roi_locations[roi_ids]
        return self.roi_locations.T

    def _create_rois_masks(self) -> _ROIMasks:
        masks = self.one.load_dataset(id=self.session, dataset="mpciROIs.masks", collection=f"alf/{self.FOV_name}")
        pixel_masks = []
        roi_id_map = {}
        cell_ids = list(range(masks.shape[0]))
        for roi_idx in cell_ids:
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

            roi_id_map[roi_idx] = roi_idx

        # Process background components if available
        if self.add_background:
            neuropil_masks = self.one.load_dataset(
                id=self.session, dataset="mpciROIs.neuropilMasks.masks", collection=f"alf/{self.FOV_name}"
            )
            background_ids = list(range(neuropil_masks.shape[0]))
            for bg_idx in background_ids:
                # Get the 2D mask for this background ROI
                bg_mask = neuropil_masks[bg_idx]  # shape: (Ly, Lx)

                # Convert to COO format to get coordinates and weights
                bg_mask_coo = sparse.COO(bg_mask)

                # Extract y, x coordinates and weights
                y_coords = bg_mask_coo.coords[0]
                x_coords = bg_mask_coo.coords[1]
                weights = bg_mask_coo.data

                # Stack into (n_pixels, 3) array: [y, x, weight]
                pixel_mask = np.vstack([y_coords, x_coords, weights]).T
                pixel_masks.append(pixel_mask)
                bg_id = f"background{bg_idx}"
                roi_id_map[bg_id] = len(pixel_masks) - 1

        self._roi_masks = _ROIMasks(
            data=pixel_masks,
            mask_tpe="nwb-pixel_mask",
            field_of_view_shape=self.get_frame_shape(),
            roi_id_map=roi_id_map,
        )
        return self._roi_masks

    def get_roi_pixel_masks(self, roi_ids=None) -> np.array:
        """Get the weights applied to each of the pixels of the mask.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested.

        Returns
        -------
        pixel_masks: list
            List of length number of rois, each element is a 2-D array with shape (number_of_non_zero_pixels, 3).
            Columns 1 and 2 are the x and y coordinates of the pixel, while the third column represents the weight of
            the pixel.
        """
        if roi_ids is None:
            roi_ids = self.get_roi_ids()

        if self._roi_masks is None:
            self._roi_masks = self._create_rois_masks()

        # Filter to only cell ROIs (exclude background)
        cell_roi_ids = [rid for rid in roi_ids if not str(rid).startswith("background")]

        # Get pixel masks from representations
        pixel_masks = []
        for roi_id in cell_roi_ids:
            pixel_mask = self._roi_masks.get_roi_pixel_mask(roi_id)
            pixel_masks.append(pixel_mask)

        return pixel_masks

    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        self._image_masks = _image_mask_extractor(
            self.get_roi_pixel_masks(roi_ids),
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
        # Determine which background ROIs to process
        if background_ids is None:
            background_ids = range(self.get_num_background_components())

        if self._roi_masks is None:
            self._roi_masks = self._create_rois_masks()

        # Filter to only background ROIs
        background_roi_ids = [rid for rid in background_ids if str(rid).startswith("background")]

        # Get pixel masks from representations
        pixel_masks = []
        for roi_id in background_roi_ids:
            pixel_mask = self._roi_masks.get_roi_pixel_mask(roi_id)
            pixel_masks.append(pixel_mask)

        return pixel_masks

    def get_background_image_masks(self, background_ids=None) -> np.ndarray:
        self._background_image_masks = _image_mask_extractor(
            self.get_background_pixel_masks(),
            background_ids if background_ids is not None else list(range(self.get_num_background_components())),
            self.get_frame_shape(),
        )
        return self._background_image_masks

    def _create_rois_responses(self) -> List[_RoiResponse]:
        """Load the ROI responses from the Suite2p output files.
        Returns
        -------
        _roi_responses: List[_RoiResponse]
            List of _RoiResponse objects containing the ROI responses.
        """
        if not self._roi_responses:
            self._roi_responses = []

            raw_traces = self.one.load_dataset(
                id=self.session, dataset="mpci.ROIActivityF", collection=f"alf/{self.FOV_name}"
            )

            if self.add_background:
                neuropil_traces = self.one.load_dataset(
                    id=self.session, dataset="mpci.ROIActivityFNeuropil", collection=f"alf/{self.FOV_name}"
                )
            else:
                neuropil_traces = None

            deconvolved_traces = self.one.load_dataset(
                id=self.session, dataset="mpci.ROIActivityDeconvolved", collection=f"alf/{self.FOV_name}"
            )

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
            self._create_rois_responses()

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
            mean_image = self.one.load_dataset(
                id=self.session, dataset="mpciMeanImage.images", collection=f"alf/{self.FOV_name}"
            )
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
