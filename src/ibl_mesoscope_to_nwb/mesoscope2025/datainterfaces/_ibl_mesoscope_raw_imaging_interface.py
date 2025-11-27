from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional

from neuroconv.datainterfaces.ophys.baseimagingextractorinterface import (
    BaseImagingExtractorInterface,
)
from neuroconv.utils import DeepDict
from pydantic import FilePath
from pynwb import NWBFile

from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import (
    IBLMesoscopeRawImagingExtractor,
)


class IBLMesoscopeRawImagingInterface(BaseImagingExtractorInterface):
    """Data Interface for IBL Mesoscope Raw Imaging data."""

    display_name = "IBL Raw Mesoscope Imaging"
    associated_suffixes = ".tif"
    info = "Interface for IBL Raw Mesoscope imaging data."

    Extractor = IBLMesoscopeRawImagingExtractor

    def __init__(
        self,
        file_path: Optional[FilePath] = None,
        channel_name: Optional[str] = None,
        file_paths: Optional[list[FilePath]] = None,
        FOV_name: str | None = None,
        plane_index: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        file_path : FilePath, optional
            Path to the ScanImage TIFF file. If this is part of a multi-file series, this should be the first file.
            Either `file_path` or `file_paths` must be provided.
        channel_name : str, optional
            Name of the channel to extract (e.g., "Channel 1", "Channel 2").

            - If None and only one channel is available, that channel will be used.
            - If None and multiple channels are available, an error will be raised.
            - Use `get_available_channels(file_path)` to see available channels before creating the interface.
        file_paths : list[Path | str], optional
            List of file paths to use. This is an escape value that can be used
            in case the automatic file detection doesn't work correctly and can be used
            to override the automatic file detection.
            This is useful when:

            - Automatic detection doesn't work correctly
            - You need to specify a custom subset of files
            - You need to control the exact order of files
            The file paths must be provided in the temporal order of the frames in the dataset.
        FOV_name : str, optional
            Name suffix to use for the imaging plane and two-photon series in the NWB file.
            If None, no suffix will be added.
        """
        file_paths = [Path(file_path)] if file_path else file_paths

        self.channel_name = channel_name
        super().__init__(
            file_path=file_path,
            channel_name=channel_name,
            file_paths=file_paths,
            plane_index=plane_index,
            verbose=verbose,
        )

        # Make sure the timestamps are available, the extractor caches them
        times = self.imaging_extractor.get_times()
        self.imaging_extractor.set_times(times=times)

        self.FOV_name = FOV_name

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

        imaging_plane_metadata.update(name=f"ImagingPlane_{self.FOV_name}")
        imaging_plane_metadata["optical_channel"].pop()  # Remove default optical channel

        two_photon_series_metadata = metadata_copy["Ophys"]["TwoPhotonSeries"][0]
        two_photon_series_metadata.update(
            name=f"TwoPhotonSeries_{self.FOV_name}",
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
