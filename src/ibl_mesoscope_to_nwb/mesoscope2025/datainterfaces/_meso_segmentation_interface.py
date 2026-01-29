from copy import deepcopy

from ibl_to_nwb.datainterfaces._base_ibl_interface import BaseIBLDataInterface
from neuroconv.datainterfaces.ophys.basesegmentationextractorinterface import (
    BaseSegmentationExtractorInterface,
)
from neuroconv.utils import DeepDict
from pydantic import DirectoryPath, validate_call
from pynwb import NWBFile

from ._meso_segmentation_extractor import MesoscopeSegmentationExtractor


class MesoscopeSegmentationInterface(BaseIBLDataInterface, BaseSegmentationExtractorInterface):
    """Interface for Meso segmentation data."""

    Extractor = MesoscopeSegmentationExtractor
    display_name = "Meso Segmentation"
    associated_suffixes = (".npy",)
    info = "Interface for Meso segmentation."

    @classmethod
    def get_extractor_class(cls):
        return MesoscopeSegmentationExtractor

    @classmethod
    def get_source_schema(cls) -> dict:
        """
        Get the source schema for the Meso segmentation interface.

        Returns
        -------
        dict
            The schema dictionary containing input parameters and descriptions
            for initializing the Meso segmentation interface.
        """
        schema = super().get_source_schema()
        schema["properties"]["folder_path"][
            "description"
        ] = "Path to the folder containing Meso segmentation data. Should contain 'FOV_#' subfolder(s)."
        schema["properties"]["FOV_name"][
            "description"
        ] = "The name of the FOV to load. This interface only loads one FOV at a time. Use the full name, e.g. 'FOV_00'. If this value is omitted, the first FOV found will be loaded."

        return schema

    @classmethod
    def get_available_planes(cls, folder_path: DirectoryPath) -> list[str]:
        """
        Get the available planes in the Meso segmentation folder.

        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the folder containing Meso segmentation data.

        Returns
        -------
        list
            List of available plane names in the dataset.
        """
        return MesoscopeSegmentationExtractor.get_available_planes(folder_path=folder_path)

    @validate_call
    def __init__(
        self,
        folder_path: DirectoryPath,
        FOV_name: str | None = None,
        verbose: bool = False,
    ):
        """

        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the folder containing Meso segmentation data. Should contain 'plane#' sub-folders.
        FOV_name: str, optional
            The name of the plane to load. This interface only loads one plane at a time.
            If this value is omitted, the first plane found will be loaded.
            To determine what planes are available, use ``MesoscopeSegmentationInterface.get_available_planes(folder_path)``.
        """

        super().__init__(folder_path=folder_path, FOV_name=FOV_name)

        self.camel_cased_FOV_name = self.segmentation_extractor.FOV_name.replace("_", "")
        self.plane_segmentation_name = f"PlaneSegmentation{self.camel_cased_FOV_name}"
        self.verbose = verbose

    def get_metadata(self) -> DeepDict:
        """
        Get metadata for the Meso segmentation data.

        Returns
        -------
        DeepDict
            Dictionary containing metadata including plane segmentation details,
            fluorescence data, and segmentation images.
        """
        metadata = super().get_metadata()
        metadata_copy = deepcopy(metadata)

        # No need to update the metadata links for the default plane segmentation name
        default_plane_segmentation_name = metadata_copy["Ophys"]["ImageSegmentation"]["plane_segmentations"][0]["name"]
        if self.plane_segmentation_name == default_plane_segmentation_name:
            return metadata_copy

        plane_segmentation_metadata = metadata_copy["Ophys"]["ImageSegmentation"]["plane_segmentations"][0]
        imaging_plane_metadata = metadata_copy["Ophys"]["ImagingPlane"][0]
        fluorescence_metadata = metadata_copy["Ophys"]["Fluorescence"]
        segmentation_images_metadata = metadata_copy["Ophys"]["SegmentationImages"]

        default_plane_segmentation_name = plane_segmentation_metadata["name"]
        imaging_plane_name = f"ImagingPlane{self.camel_cased_FOV_name}"

        plane_segmentation_metadata.update(
            name=self.plane_segmentation_name,
            imaging_plane=imaging_plane_name,
        )

        imaging_plane_metadata.update(name=imaging_plane_name)
        imaging_plane_metadata["optical_channel"].pop()  # Remove default optical channel

        fluorescence_metadata_per_plane = fluorescence_metadata.pop(default_plane_segmentation_name)
        # override the default name of the plane segmentation
        fluorescence_metadata[self.plane_segmentation_name] = fluorescence_metadata_per_plane
        trace_names = [
            property_name for property_name in fluorescence_metadata_per_plane.keys() if property_name != "name"
        ]
        for trace_name in trace_names:
            fluorescence_metadata_per_plane[trace_name].update(
                name=trace_name.upper() + f"ROIResponseSeries{self.camel_cased_FOV_name}"
            )

        segmentation_images_metadata_per_plane = segmentation_images_metadata.pop(default_plane_segmentation_name)
        segmentation_images_metadata[self.plane_segmentation_name] = segmentation_images_metadata_per_plane
        segmentation_images_metadata[self.plane_segmentation_name].update(
            correlation=dict(name=f"CorrelationImage{self.camel_cased_FOV_name}"),
            mean=dict(name=f"MeanImage{self.camel_cased_FOV_name}"),
        )

        return metadata_copy

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict | None = None,
        stub_test: bool = False,
        stub_samples: int = 100,
        include_roi_centroids: bool = True,
        include_roi_acceptance: bool = True,
        mask_type: str = "pixel",  # Literal["image", "pixel", "voxel"]
        plane_segmentation_name: str | None = None,
        iterator_options: dict | None = None,
    ):
        """
        Add segmentation data to the specified NWBFile.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWBFile object to which the segmentation data will be added.
        metadata : dict, optional
            Metadata containing information about the segmentation. If None, default metadata is used.
        stub_test : bool, optional
            If True, only a subset of the data (defined by `stub_samples`) will be added for testing purposes,
            by default False.
        stub_samples : int, optional
            The number of samples to include in the subset if `stub_test` is True, by default 100.
        include_roi_centroids : bool, optional
            Whether to include the centroids of regions of interest (ROIs) in the data, by default True.
        include_roi_acceptance : bool, optional
            Whether to include acceptance status of ROIs, by default True.
        mask_type : str, default: 'image'
            There are three types of ROI masks in NWB, 'image', 'pixel', and 'voxel'.

            * 'image' masks have the same shape as the reference images the segmentation was applied to, and weight each pixel
            by its contribution to the ROI (typically boolean, with 0 meaning 'not in the ROI').
            * 'pixel' masks are instead indexed by ROI, with the data at each index being the shape of the image by the number
            of pixels in each ROI.
            * 'voxel' masks are instead indexed by ROI, with the data at each index being the shape of the volume by the number
            of voxels in each ROI.

            Specify your choice between these two as mask_type='image', 'pixel', 'voxel', or None.
            plane_segmentation_name : str, optional
            The name of the plane segmentation object, by default None.
        iterator_options : dict, optional
            Additional options for iterating over the data, by default None.
        """
        super().add_to_nwbfile(
            nwbfile=nwbfile,
            metadata=metadata,
            stub_test=stub_test,
            stub_samples=stub_samples,
            include_roi_centroids=include_roi_centroids,
            include_roi_acceptance=include_roi_acceptance,
            mask_type=mask_type,
            plane_segmentation_name=self.plane_segmentation_name,
            iterator_options=iterator_options,
        )
