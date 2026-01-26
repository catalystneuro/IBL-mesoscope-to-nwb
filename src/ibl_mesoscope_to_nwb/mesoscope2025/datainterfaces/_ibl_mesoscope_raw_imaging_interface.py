import json
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from neuroconv.datainterfaces.ophys.baseimagingextractorinterface import (
    BaseImagingExtractorInterface,
)
from neuroconv.utils import DeepDict, dict_deep_update, load_dict_from_file
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

    def __init__(
        self,
        file_path: Optional[FilePath] = None,
        channel_name: Optional[str] = None,
        file_paths: Optional[list[FilePath]] = None,
        FOV_name: str | None = None,
        task: str = "",
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

        self.raw_imaging_metadata_path = file_paths[0].parent.parent / "_ibl_rawImagingData.meta.json"

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
        self.FOV_index = plane_index if plane_index is not None else 0
        self.task = task

    def get_extractor_class(self):
        return IBLMesoscopeRawImagingExtractor

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

        metadata_copy["Ophys"]["ImagingPlane"][0]["optical_channel"].pop()  # Remove default optical channel

        # Use single source of truth when updating metadata
        ophys_metadata = load_dict_from_file(
            file_path=Path(__file__).parent.parent / "_metadata" / "mesoscope_raw_ophys_metadata.yaml"
        )

        with open(self.raw_imaging_metadata_path, "r") as f:
            raw_metadata = json.load(f)
            fov = raw_metadata["FOV"][self.FOV_index]

        two_photon_series_suffix = self.FOV_name.replace("_", "") + self.task
        imaging_plane_suffix = self.FOV_name.replace("_", "")
        # Get the template structures (single entries from YAML)
        imaging_plane_template = ophys_metadata["Ophys"]["ImagingPlane"][0]
        two_photon_series_template = ophys_metadata["Ophys"]["TwoPhotonSeries"][0]

        # Get global imaging rate
        imaging_rate = raw_metadata["scanImageParams"]["hRoiManager"]["scanFrameRate"]
        scan_line_rate = raw_metadata["scanImageParams"]["hRoiManager"]["linePeriod"] ** -1

        # Get device information (assumed single device)
        device_metadata = ophys_metadata["Ophys"]["Device"][0]

        # Extract FOV-specific metadata
        fov_uuid = fov["roiUUID"]

        dimensions = fov["nXnYnZ"]  # [width, height, depth] in pixels

        x_pixel_size = raw_metadata["rawScanImageMeta"]["XResolution"]  # in micrometers
        y_pixel_size = raw_metadata["rawScanImageMeta"]["YResolution"]  # in micrometers

        grid_spacing = [x_pixel_size * 1e-6, y_pixel_size * 1e-6]  # x spacing in meters  # y spacing in meters

        # Create ImagingPlane entry for this FOV
        imaging_plane = imaging_plane_template.copy()
        imaging_plane["name"] = f"ImagingPlane{imaging_plane_suffix}"
        imaging_plane["description"] = (
            f"Field of view {self.FOV_index} (UUID: {fov_uuid}). "
            f"Image dimensions: {dimensions[0]}x{dimensions[1]} pixels."
        )
        imaging_plane["imaging_rate"] = imaging_rate
        if "brainLocationIds" in fov:
            brain_region_id = fov["brainLocationIds"]["center"]
            imaging_plane["location"] = (
                f"Brain region ID {brain_region_id} (Allen CCF 2017)" if brain_region_id is not None else "Unknown"
            )
        imaging_plane["grid_spacing"] = grid_spacing
        imaging_plane["device"] = device_metadata["name"]

        # Create TwoPhotonSeries entry for this FOV
        two_photon_series = two_photon_series_template.copy()
        two_photon_series["name"] = f"TwoPhotonSeries{two_photon_series_suffix}"
        two_photon_series["description"] = (
            f"The raw two-photon imaging data acquired using the mesoscope on {self.FOV_name} (UUID: {fov_uuid}) ."
        )
        two_photon_series["imaging_plane"] = f"ImagingPlane{imaging_plane_suffix}"
        two_photon_series["scan_line_rate"] = scan_line_rate

        metadata_copy["Ophys"]["Device"][0] = device_metadata
        metadata_copy["Ophys"]["ImagingPlane"][0] = dict_deep_update(
            metadata_copy["Ophys"]["ImagingPlane"][0], imaging_plane
        )
        metadata_copy["Ophys"]["TwoPhotonSeries"][0] = dict_deep_update(
            metadata_copy["Ophys"]["TwoPhotonSeries"][0], two_photon_series
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
