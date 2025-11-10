from copy import deepcopy
from typing import Literal

from neuroconv.datainterfaces.ophys.baseimagingextractorinterface import (
    BaseImagingExtractorInterface,
)
from neuroconv.utils import DeepDict
from pydantic import FilePath
from pynwb import NWBFile

from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import (
    MotionCorrectedMesoscopeImagingExtractor,
)


class MotionCorrectedMesoscopeImagingInterface(BaseImagingExtractorInterface):
    """Data Interface for MotionCorrectedMesoscopeImagingExtractor."""

    display_name = "IBL Motion Corrected Mesoscope Imaging"
    associated_suffixes = (".bin", ".npy")
    info = "Interface for IBL Motion Corrected Mesoscope imaging data."

    Extractor = MotionCorrectedMesoscopeImagingExtractor

    def __init__(
        self,
        file_path: FilePath,
        verbose: bool = False,
    ):

        # Validate file path structure
        if not file_path.name == "imaging.frames_motionRegistered.bin":
            raise ValueError(f"Expected file named 'imaging.frames_motionRegistered.bin', got '{file_path.name}'")

        super().__init__(
            file_path=file_path,
            verbose=verbose,
        )
        self.two_photon_series_name_suffix = self.imaging_extractor._alf_folder.name

    def get_metadata(self) -> DeepDict:
        """
        Get metadata for the Miniscope imaging data.

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
        parent_container: Literal["acquisition"] | Literal["processing/ophys"] = "processing/ophys",
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
