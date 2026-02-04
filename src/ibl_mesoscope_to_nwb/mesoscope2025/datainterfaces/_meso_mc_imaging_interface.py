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
    MesoscopeMotionCorrectedImagingExtractor,
)
from ibl_mesoscope_to_nwb.mesoscope2025.utils import get_FOV_names_from_alf_collections


class MesoscopeMotionCorrectedImagingInterface(BaseIBLDataInterface, BaseImagingExtractorInterface):
    """Data Interface for MesoscopeMotionCorrectedImagingExtractor."""

    display_name = "IBL Motion Corrected Mesoscope Imaging"
    info = "Interface for IBL Motion Corrected Mesoscope imaging data."

    @classmethod
    def get_extractor_class(cls):
        return MesoscopeMotionCorrectedImagingExtractor

    def __init__(self, one: ONE, session: str, FOV_name: str, verbose: bool = True):
        self.one = one
        self.session = session
        self.revision = self.REVISION
        # Check if task exists
        FOV_names = get_FOV_names_from_alf_collections(one, session)
        if FOV_name not in FOV_names:
            raise ValueError(
                f"FOV_name '{FOV_name}' not found for session '{session}'. " f"Available FOV_names: {FOV_names}.'"
            )
        super().__init__(one=one, session=session, FOV_name=FOV_name)
        self.FOV_name = FOV_name
        self.FOV_index = int(FOV_name.replace("FOV_", ""))
        self.verbose = verbose

    @classmethod
    def get_data_requirements(cls, FOV_name: str) -> dict:
        """
        Declare exact data files required for anatomical localization.

        Note: This interface derives anatomical localization from specific numpy files.

        Returns
        -------
        dict
            Data requirements with exact file paths
        """
        FOV_index = int(FOV_name.replace("FOV_", ""))
        plane_name = f"plane{FOV_index}"
        return {
            "exact_files_options": {
                "standard": [
                    f"alf/{FOV_name}/mpci.times.npy",
                    f"suite2p/{plane_name}/imaging.frames_motionRegistered.bin",
                ]
            },
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
        Get metadata for the Meso segmentation data.

        Returns
        -------
        DeepDict
            Dictionary containing metadata including plane segmentation details,
            fluorescence data, and segmentation images.
        """
        metadata = super().get_metadata()
        metadata_copy = deepcopy(metadata)  # To avoid modifying the parent class's metadata

        metadata_copy["Ophys"]["ImagingPlane"][0]["optical_channel"].pop()  # Remove default optical channel

        # Use single source of truth when updating metadata
        ophys_metadata = load_dict_from_file(
            file_path=Path(__file__).parent.parent / "_metadata" / "mesoscope_processed_ophys_metadata.yaml"
        )
        raw_metadata = self.one.load_dataset(
            self.session, dataset="_ibl_rawImagingData.meta.json", collection="raw_imaging_data_00"
        )
        fov = raw_metadata["FOV"][self.FOV_index]

        suffix = self.FOV_name.replace("_", "")

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
        imaging_plane["name"] = f"ImagingPlane{suffix}"
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
        two_photon_series["name"] = f"MotionCorrectedTwoPhotonSeries{suffix}"
        two_photon_series["description"] = (
            f"The motion corrected two-photon imaging data acquired using the mesoscope on {self.FOV_name} (UUID: {fov_uuid})."
        )
        two_photon_series["imaging_plane"] = f"ImagingPlane{suffix}"
        two_photon_series["scan_line_rate"] = scan_line_rate

        # Create Device entry
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
