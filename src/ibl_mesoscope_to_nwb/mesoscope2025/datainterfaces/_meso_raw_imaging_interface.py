import tarfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Literal

from ibl_to_nwb.datainterfaces._base_ibl_interface import BaseIBLDataInterface
from neuroconv.datainterfaces.ophys.baseimagingextractorinterface import (
    BaseImagingExtractorInterface,
)
from neuroconv.utils import DeepDict, dict_deep_update, load_dict_from_file
from one.api import ONE
from pynwb import NWBFile

from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import (
    MesoscopeRawImagingExtractor,
)
from ibl_mesoscope_to_nwb.mesoscope2025.utils import (
    get_number_of_FOVs_from_raw_imaging_metadata,
)


class MesoscopeRawImagingInterface(BaseIBLDataInterface, BaseImagingExtractorInterface):
    """Data Interface for IBL Mesoscope Raw Imaging data."""

    display_name = "IBL Raw Mesoscope Imaging"
    info = "Interface for IBL Raw Mesoscope imaging data."

    @classmethod
    def get_extractor_class(cls):
        return MesoscopeRawImagingExtractor

    def __init__(
        self,
        one: ONE,
        session: str,
        FOV_index: int,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        one : ONE
            An instance of the ONE API.
        session : str
            The session ID for the experiment.
        FOV_index : int
            Index of the field of view (FOV) for the imaging data.
        verbose : bool, optional
            If True, enables verbose output during processing.
        """
        self.one = one
        self.session = session
        assert FOV_index < get_number_of_FOVs_from_raw_imaging_metadata(
            self.one, self.session
        ), f"FOV_index {FOV_index} is out of range for session {self.session}."
        self.FOV_name = f"FOV_{FOV_index:02d}"
        self.FOV_index = FOV_index

        channel_name = "OpticalChannel"

        super().__init__(
            one=one,
            session=session,
            FOV_index=FOV_index,
            channel_name=channel_name,
            verbose=verbose,
        )

    @classmethod
    def get_data_requirements(cls) -> dict:
        """
        Declare exact data files required for raw imaging data.

        Note: This interface derives raw imaging data from specific numpy files.

        Returns
        -------
        dict
            Data requirements with exact file paths
        """
        # TODO - is the raw_imaging_data_00 collection necessary and sufficient condition?
        return {
            "exact_files_options": {
                "standard": [
                    f"raw_imaging_data_00/imaging.frames.tar.bz2",
                    f"raw_imaging_data_00/rawImagingData.times_scanImage.npy",
                ]
            },
        }

    @classmethod
    def download_data(
        cls,
        one: ONE,
        eid: str,
        decompress: bool = True,
        download_only: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> dict:
        """
        Download raw imaging data for all imaging blocks in the session and optionally decompress.

        Downloads all raw_imaging_data_* collections (one per imaging block). If decompress=True,
        each imaging.frames.tar.bz2 is extracted in-place after download.

        Parameters
        ----------
        one : ONE
            ONE API instance
        eid : str
            Session ID
        decompress : bool, default=True
            If True, extract imaging.frames.tar.bz2 files after downloading
        download_only : bool, default=True
            If True, download files but don't load into memory
        verbose : bool, default=False
            If True, print download status and timing information

        Returns
        -------
        dict
            Download status
        """
        requirements = cls.get_data_requirements()

        start_time = time.time()

        # Discover all raw_imaging_data_* collections (excludes reference)
        all_datasets = one.list_datasets(eid)
        raw_imaging_collections = sorted(
            set(
                str(d).split("/")[0]
                for d in all_datasets
                if str(d).startswith("raw_imaging_data") and "reference" not in str(d)
            )
        )

        downloaded_files = []
        for collection in raw_imaging_collections:
            # NO try-except - let it fail if files missing
            one.load_object(eid, obj="imaging", collection=collection, download_only=download_only)
            one.load_object(eid, obj="rawImagingData", collection=collection, download_only=download_only)
            downloaded_files.extend(
                [f"{collection}/imaging.frames.tar.bz2", f"{collection}/rawImagingData.times_scanImage.npy"]
            )

        # Decompress tar.bz2 files after download
        if decompress:
            session_path = one.eid2path(eid)
            for collection in raw_imaging_collections:
                tar_path = Path(session_path) / collection / "imaging.frames.tar.bz2"
                if tar_path.exists():
                    extract_path = tar_path.parent / "imaging.frames"
                    if extract_path.exists() and any(extract_path.iterdir()):
                        if verbose:
                            print(f"  Skipping decompression, already extracted: {extract_path}")
                        continue
                    if verbose:
                        print(f"  Decompressing {tar_path}...")
                    extract_path.mkdir(exist_ok=True)
                    with tarfile.open(tar_path, "r:*") as tar:
                        tar.extractall(path=extract_path)

        download_time = time.time() - start_time

        if verbose:
            print(
                f"  Downloaded{' and decompressed' if decompress else ''} raw imaging data "
                f"({len(raw_imaging_collections)} collections) in {download_time:.2f}s"
            )

        return {
            "success": True,
            "downloaded_objects": ["imaging", "rawImagingData"],
            "downloaded_files": requirements["exact_files_options"]["standard"],
            "already_cached": [],
            "alternative_used": None,
            "data": None,
        }

    @classmethod
    def check_availability(cls, one: ONE, eid: str, **kwargs) -> dict:
        """
        Check if required data is available for a specific session.

        This method NEVER downloads data - it only checks if files exist
        using one.list_datasets(). It's designed to be fast and read-only,
        suitable for scanning many sessions.

        NO try-except patterns that hide failures. If checking fails,
        let the exception propagate.

        NOTE: Does NOT use revision filtering in check_availability(). Queries for latest
        version of all files regardless of revision tags. This matches the smart fallback
        behavior of load_object() and download methods, which try requested revision first
        but fall back to latest if not found.

        Parameters
        ----------
        one : ONE
            ONE API instance
        eid : str
            Session ID (experiment ID)
        **kwargs : dict
            Interface-specific parameters

        Returns
        -------
        dict
            {
                "available": bool,              # Overall availability
                "missing_required": [str],      # Missing required files
                "found_files": [str],           # Files that exist
                "alternative_used": str,        # Which alternative was found (if applicable)
                "requirements": dict,           # Copy of get_data_requirements()
            }

        Examples
        --------
        >>> result = WheelInterface.check_availability(one, eid)
        >>> if not result["available"]:
        >>>     print(f"Missing: {result['missing_required']}")
        """
        # STEP 1: Check quality (QC filtering)
        quality_result = cls.check_quality(one=one, eid=eid, **kwargs)

        if quality_result is not None:
            # If quality check explicitly rejects, return immediately
            if quality_result.get("available") is False:
                return quality_result
            # Otherwise, save extra fields to merge later
            extra_fields = quality_result
        else:
            extra_fields = {}

        # STEP 2: Check file existence
        requirements = cls.get_data_requirements(**kwargs)

        # Query without revision filtering to get latest version of ALL files
        # This includes both revision-tagged files (spike sorting) and untagged files (behavioral)
        # The unfiltered query returns the superset of what any revision-specific query would return
        available_datasets = one.list_datasets(eid)
        available_files = set(str(d) for d in available_datasets)

        missing_required = []
        found_files = []
        alternative_used = None

        # Check file options - this is now REQUIRED (not optional)
        # Every interface must define exact_files_options dict
        exact_files_options = requirements.get("exact_files_options", {})

        if not exact_files_options:
            raise ValueError(
                f"{cls.__name__}.get_data_requirements() must return 'exact_files_options' dict. "
                f"Even for single-format interfaces, use: {{'standard': ['file1.npy', 'file2.npy']}}"
            )

        # Check each named option - ANY complete option = available
        for option_name, option_files in exact_files_options.items():
            all_files_found = True

            for exact_file in option_files:
                # Handle wildcards
                if "*" in exact_file:
                    import re

                    pattern = re.escape(exact_file).replace(r"\*", ".*")
                    found = any(re.search(pattern, avail) for avail in available_files)
                else:
                    found = any(exact_file in avail for avail in available_files)

                if not found:
                    all_files_found = False
                    break  # This option is incomplete

            # If this option has all files, mark as available
            if all_files_found:
                found_files.extend(option_files)
                alternative_used = option_name  # Report which option was found
                break  # Found one complete option, that's enough

        # If no options were complete, mark the first option as missing for reporting
        if not alternative_used:
            first_option_name = next(iter(exact_files_options.keys()))
            missing_required.extend(exact_files_options[first_option_name])

        # STEP 3: Build result and merge extra fields from quality check
        result = {
            "available": len(missing_required) == 0,
            "missing_required": missing_required,
            "found_files": found_files,
            "alternative_used": alternative_used,
            "requirements": requirements,
        }
        result.update(extra_fields)

        return result

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

        raw_metadata = self.one.load_dataset(
            self.session, dataset="_ibl_rawImagingData.meta.json", collection=f"raw_imaging_data_00"
        )
        fov = raw_metadata["FOV"][self.FOV_index]

        two_photon_series_suffix = self.FOV_name.replace("_", "")
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
