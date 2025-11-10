from copy import deepcopy
from typing import Literal, Optional

from neuroconv.utils import DeepDict
from pydantic import FilePath
from pynwb import NWBFile

from neuroconv.datainterfaces import ScanImageImagingInterface


class IBLMesoscopeRawImagingInterface(ScanImageImagingInterface):
    """Data Interface for IBL Mesoscope Raw Imaging data."""

    display_name = "IBL Raw Mesoscope Imaging"
    associated_suffixes = ".tif"
    info = "Interface for IBL Raw Mesoscope imaging data."

    ExtractorName = "ScanImageImagingExtractor"

    def __init__(
        self,
        file_path: Optional[FilePath] = None,
        channel_name: Optional[str] = None,
        slice_sample: Optional[int] = None,
        plane_index: Optional[int] = None,
        file_paths: Optional[list[FilePath]] = None,
        interleave_slice_samples: Optional[bool] = None,
        plane_name: str | None = None,
        fallback_sampling_frequency: float | None = None,
        verbose: bool = False,
    ):

        super().__init__(
            file_path=file_path,
            channel_name=channel_name,
            slice_sample=slice_sample,
            plane_index=plane_index,
            file_paths=file_paths,
            interleave_slice_samples=interleave_slice_samples,
            fallback_sampling_frequency=fallback_sampling_frequency,
            verbose=verbose,
        )
        self.two_photon_series_name_suffix = plane_name

    def get_metadata(self) -> DeepDict:
        """
        Get metadata for the IBL imaging data.

        Returns
        -------
        DeepDict
            Dictionary containing metadata including device information, imaging plane details,
            and one-photon series configuration.
        """
        metadata = super().get_metadata()
        metadata_copy = deepcopy(metadata)  # To avoid modifying the parent class's metadata
        imaging_plane_metadata = metadata_copy["Ophys"]["ImagingPlane"][0]
        two_photon_series_metadata = metadata_copy["Ophys"]["TwoPhotonSeries"][0]

        imaging_plane_metadata.update(name=f"imaging_plane_{self.two_photon_series_name_suffix}")
        imaging_plane_metadata["optical_channel"].pop()  # Remove default optical channel

        two_photon_series_metadata = metadata_copy["Ophys"]["TwoPhotonSeries"][0]
        two_photon_series_metadata.update(
            name=f"two_photon_series_{self.two_photon_series_name_suffix}",
            imaging_plane=imaging_plane_metadata["name"],
        )

        return metadata_copy

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict | None = None,
        photon_series_type: Literal["TwoPhotonSeries"] | Literal["OnePhotonSeries"] = "TwoPhotonSeries",
        photon_series_index: int = 0,
        parent_container: Literal["acquisition"] | Literal["processing/ophys"] = "acquisition",
        stub_test: bool = False,
        stub_frames: int | None = None,
        always_write_timestamps: bool = True,
        iterator_type: str | None = "v2",
        iterator_options: dict | None = None,
        stub_samples: int = 100,
    ):
        return super().add_to_nwbfile(
            nwbfile,
            metadata,
            photon_series_type,
            photon_series_index,
            parent_container,
            stub_test,
            stub_frames,
            always_write_timestamps,
            iterator_type,
            iterator_options,
            stub_samples,
        )
