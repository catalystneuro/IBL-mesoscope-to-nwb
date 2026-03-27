import time
from typing import Optional

import numpy as np
from ibl_to_nwb.datainterfaces._base_ibl_interface import BaseIBLDataInterface
from iblatlas.atlas import MRITorontoAtlas
from ndx_anatomical_localization import (
    AllenCCFv3Space,
    AnatomicalCoordinatesImage,
    AnatomicalCoordinatesTable,
    Localization,
    Space,
)
from one.api import ONE
from pynwb import NWBFile
from pynwb.ophys import ImageSegmentation

from ibl_mesoscope_to_nwb.mesoscope2025.utils import get_FOV_names_from_alf_collections


class MesoscopeROIAnatomicalLocalizationInterface(BaseIBLDataInterface):
    """An anatomical localization interface for segmented ROIs."""

    interface_name = "MesoscopeROIAnatomicalLocalizationInterface"
    REVISION: str | None = None

    def __init__(self, one: ONE, session: str, FOV_name: str):
        """Initialize the ROI anatomical localization interface.

        Validates that the given FOV_name exists for the session and sets up
        IBL-Bregma and Allen CCF v3 coordinate space objects.

        Parameters
        ----------
        one : ONE
            ONE API instance for data access.
        session : str
            Session ID (experiment UUID / eid).
        FOV_name : str
            Field of view name (e.g. "FOV_00", "FOV_01"). Must match an entry
            returned by `get_FOV_names_from_alf_collections`.

        Raises
        ------
        ValueError
            If `FOV_name` is not found in the session's ALF collections.
        """
        self.one = one
        self.session = session
        self.revision = self.REVISION
        # Check if FOV_name exists
        FOV_names = get_FOV_names_from_alf_collections(one, session)
        if FOV_name not in FOV_names:
            raise ValueError(
                f"FOV_name '{FOV_name}' not found for session '{session}'. " f"Available FOV_names: {FOV_names}.'"
            )
        self.FOV_name = FOV_name
        self.camel_case_FOV_name = FOV_name.replace("_", "")
        # IBL bregma-centred space: origin = bregma, units = um, orientation = RAS
        #   x = ML (mediolateral, +right), y = AP (anteroposterior, +anterior), z = DV (+dorsal)
        self.ibl_bregma_space = Space(
            name="IBLBregma",
            space_name="IBLBregma",
            origin="bregma",
            units="um",
            orientation="RAS",
        )
        self.allen_ccf_space = AllenCCFv3Space()  # standard Allen CCF v3 space (PIR+ orientation)
        self.atlas = MRITorontoAtlas(res_um=10)  # The MRI Toronto brain atlas
        super().__init__(one=one, session=session)

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
        return {
            "exact_files_options": {
                "standard": [
                    f"alf/{FOV_name}/mpciROIs.mlapdv_estimate.npy",
                    f"alf/{FOV_name}/mpciROIs.brainLocationIds_ccf_2017_estimate.npy",
                ]
            },
        }

    # WARNING: The following method cannot be staticmethod due to self.FOV_name usage
    def get_load_object_kwargs(self) -> dict:
        """Return kwargs for one.load_object() call."""
        return {"obj": "mpciROIs", "collection": f"alf/{self.FOV_name}"}

    @classmethod
    def download_data(
        cls,
        one: ONE,
        eid: str,
        FOV_name: str,
        download_only: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> dict:
        """
        Download anatomical localization data.

        NOTE: Uses class-level REVISION attribute automatically.

        Parameters
        ----------
        one : ONE
            ONE API instance
        eid : str
            Session ID
        FOV_name : str
            Field of view name (e.g. "FOV_00", "FOV_01", etc.) to specify which ROI localization data to download
        download_only : bool, default=True
            If True, download but don't load into memory
        verbose : bool, default=False
            If True, print download status and timing information

        Returns
        -------
        dict
            Download status
        """
        requirements = cls.get_data_requirements(
            FOV_name=FOV_name
        )  # pass FOV_name from kwargs to get_data_requirements

        # Use class-level REVISION attribute
        revision = cls.REVISION

        start_time = time.time()

        # NO try-except - let it fail if files missing
        one.load_object(
            eid, obj="mpciROIs", collection=f"alf/{FOV_name}", download_only=download_only, revision=revision
        )

        download_time = time.time() - start_time

        if verbose:
            print(f"  Downloaded ROI localization data for {FOV_name} in {download_time:.2f}s")

        # SessionLoader handles format detection internally, report BWM format as default
        # (it will fall back to legacy if needed)
        return {
            "success": True,
            "downloaded_objects": ["mpciROIs"],
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

    def _add_coordinate_spaces(self, nwbfile: NWBFile):
        """Add coordinate spaces to the NWB file.

        Creates Space objects for the IBL bregma-centered coordinate system and the Allen CCF space,
        and adds them to a Localization container in the NWB file.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to which the coordinate spaces will be added.
        """

        if "localization" not in nwbfile.lab_meta_data:  # create Localization container if missing
            nwbfile.add_lab_meta_data([Localization()])

        localization = nwbfile.lab_meta_data["localization"]
        if (
            self.ibl_bregma_space.name not in localization.spaces
            and self.allen_ccf_space.name not in localization.spaces
        ):
            localization.add_spaces([self.ibl_bregma_space, self.allen_ccf_space])  # register both coordinate spaces

    def _ensure_plane_segmentation_exists(self, nwbfile: NWBFile):
        """Retrieve the PlaneSegmentation for this FOV from the NWB file.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to search.

        Returns
        -------
        PlaneSegmentation
            The plane segmentation table for this FOV.

        Raises
        ------
        ValueError
            If the 'ophys' processing module, ImageSegmentation container, or
            the PlaneSegmentation for this FOV does not exist or is empty.
        """
        if "ophys" not in nwbfile.processing:
            raise ValueError("No 'ophys' processing module found in NWB file.")

        segmentation_module = None
        for name, proc in nwbfile.processing["ophys"].data_interfaces.items():
            if isinstance(proc, ImageSegmentation):
                segmentation_module = nwbfile.processing["ophys"][name]
                break

        if segmentation_module is None:
            raise ValueError("No ImageSegmentation data interface found in 'ophys' processing module.")

        plane_segmentation = None
        if segmentation_module is not None:
            for ps_name, ps_object in segmentation_module.plane_segmentations.items():
                if self.camel_case_FOV_name in ps_name:
                    plane_segmentation = ps_object
                    break
        if plane_segmentation is None:
            raise ValueError(
                f"Plane segmentation for {self.FOV_name} doesn't exist. "
                "Populate the plane segmentation table first "
                "(e.g. via MesoscopeSegmentationInterface in the processed pipeline) "
                "before running the anatomical localization interface."
            )
        if len(plane_segmentation) == 0:
            raise ValueError(
                f"Plane segmentation for {self.FOV_name} is empty. "
                "Populate the plane segmentation table first "
                "(e.g. via MesoscopeSegmentationInterface in the processed pipeline) "
                "before running the anatomical localization interface."
            )
        return plane_segmentation

    def _build_anatomical_coordinates_table(self, nwbfile: NWBFile):
        """Build AnatomicalCoordinatesTable objects for IBL-Bregma and Allen CCF v3 spaces.

        Loads ROI coordinates from `mpciROIs.mlapdv_estimate` (ML, AP, DV in µm,
        IBL-Bregma RAS space) and brain region IDs from
        `mpciROIs.brainLocationIds_ccf_2017_estimate`, then converts IBL-Bregma
        coordinates to Allen CCF v3 (AP, DV, ML in µm) using the MRI Toronto atlas.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file containing the target PlaneSegmentation.

        Returns
        -------
        tuple[AnatomicalCoordinatesTable, AnatomicalCoordinatesTable]
            A pair of tables: (IBL-Bregma table, Allen CCF v3 table).
            Each row corresponds to one ROI in the PlaneSegmentation.
        """
        # Get anatomical localization data
        rois = self.one.load_object(self.session, **self.get_load_object_kwargs())
        rois_mlapdv = rois["mlapdv_estimate"]
        rois_brain_location_ids = rois["brainLocationIds_ccf_2017_estimate"]

        plane_segmentation = self._ensure_plane_segmentation_exists(nwbfile)

        # Precompute brain region name lookup: id -> name (using Allen CCF 2017 atlas)
        unique_ids = np.unique(rois_brain_location_ids)
        region_info = self.atlas.regions.get(unique_ids)
        id_to_acronym = dict(zip(unique_ids.tolist(), region_info.acronym.tolist()))

        # Create AnatomicalCoordinatesTable for IBL-Bregma coordinates
        ibl_anatomical_coordinates_table = AnatomicalCoordinatesTable(
            name=f"AnatomicalCoordinatesTableIBLBregmaROI{self.camel_case_FOV_name}",
            description=f"ROI centroid estimated coordinates in the IBL-Bregma coordinate system for {self.FOV_name}.",
            target=plane_segmentation,
            space=nwbfile.lab_meta_data["localization"].spaces[self.ibl_bregma_space.name],
            method="<>",  # TODO Add method description
        )
        ibl_anatomical_coordinates_table.add_column(
            name="brain_region_id",
            description="The brain region IDs are from the 2017 Allen CCF atlas.",
        )

        # Create AnatomicalCoordinatesTable for CCF coordinates
        ccf_anatomical_coordinates_table = AnatomicalCoordinatesTable(
            name=f"AnatomicalCoordinatesTableCCFv3ROI{self.camel_case_FOV_name}",
            description=f"ROI centroid estimated coordinates in the CCF coordinate system for {self.FOV_name}.",
            target=plane_segmentation,
            space=nwbfile.lab_meta_data["localization"].spaces[self.allen_ccf_space.name],
            method="<>",  # TODO Add method description
        )
        ccf_anatomical_coordinates_table.add_column(
            name="brain_region_id",
            description="The brain region IDs are from the 2017 Allen CCF atlas.",
        )

        # Populate tables with coordinates for each ROI
        for roi_index in plane_segmentation.id[:]:
            x = float(rois_mlapdv[roi_index][0])
            y = float(rois_mlapdv[roi_index][1])
            z = float(rois_mlapdv[roi_index][2])
            brain_region_id = int(rois_brain_location_ids[roi_index])
            brain_region_acronym = id_to_acronym[brain_region_id]
            ibl_anatomical_coordinates_table.add_row(
                localized_entity=roi_index,
                x=x,
                y=y,
                z=z,
                brain_region_id=brain_region_id,
                brain_region=brain_region_acronym,
            )
            xyz_m_for_ccf = np.hstack((x, -y, z)) / 1e6  # convert from um to m and switch to PIR+ for CCF
            ccf_um = self.atlas.xyz2ccf(xyz=xyz_m_for_ccf, ccf_order="apdvml").astype(np.float64)  # shape (N, 3)
            ccf_anatomical_coordinates_table.add_row(
                localized_entity=roi_index,
                x=ccf_um[0],
                y=ccf_um[1],
                z=ccf_um[2],
                brain_region_id=brain_region_id,
                brain_region=brain_region_acronym,
            )

        return ibl_anatomical_coordinates_table, ccf_anatomical_coordinates_table

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: Optional[dict] = None):
        """
        Add anatomical localization data to the NWB file.

        This method ONLY adds AnatomicalCoordinatesTable objects linking segmented ROIs
        to IBL-Bregma coordinate systems and CCF brain region IDS. The plane segmentation tables must already
        exist with anatomical columns populated (done by MesoscopeSegmentationInterface).

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to add data to
        metadata : dict, optional
            Metadata dictionary (not currently used)

        Raises
        ------
        ValueError
            If plane segmentation table doesn't exist or is missing required columns
        """
        self._add_coordinate_spaces(nwbfile)
        localization = nwbfile.lab_meta_data["localization"]

        # Build AnatomicalCoordinatesTable
        ibl_anatomical_coordinates_table, ccf_anatomical_coordinates_table = self._build_anatomical_coordinates_table(
            nwbfile=nwbfile
        )
        localization.add_anatomical_coordinates_tables(
            [ibl_anatomical_coordinates_table, ccf_anatomical_coordinates_table]
        )


class MesoscopeImageAnatomicalLocalizationInterface(BaseIBLDataInterface):
    """An anatomical localization interface for Mean Projection."""

    interface_name = "MesoscopeImageAnatomicalLocalizationInterface"

    REVISION: str | None = None

    def __init__(self, one: ONE, session: str, FOV_name: str):
        """Initialize the mean image anatomical localization interface.

        Validates that the given FOV_name exists for the session and sets up
        IBL-Bregma and Allen CCF v3 coordinate space objects.

        Parameters
        ----------
        one : ONE
            ONE API instance for data access.
        session : str
            Session ID (experiment UUID / eid).
        FOV_name : str
            Field of view name (e.g. "FOV_00", "FOV_01"). Must match an entry
            returned by `get_FOV_names_from_alf_collections`.

        Raises
        ------
        ValueError
            If `FOV_name` is not found in the session's ALF collections.
        """
        self.one = one
        self.session = session
        self.revision = self.REVISION
        # Check if FOV_names exists
        FOV_names = get_FOV_names_from_alf_collections(one, session)
        if FOV_name not in FOV_names:
            raise ValueError(
                f"FOV_name '{FOV_name}' not found for session '{session}'. " f"Available FOV_names: {FOV_names}.'"
            )
        self.FOV_name = FOV_name
        self.camel_case_FOV_name = FOV_name.replace("_", "")
        # IBL bregma-centred space: origin = bregma, units = um, orientation = RAS
        #   x = ML (mediolateral, +right), y = AP (anteroposterior, +anterior), z = DV (+dorsal)
        self.ibl_bregma_space = Space(
            name="IBLBregma",
            space_name="IBLBregma",
            origin="bregma",
            units="um",
            orientation="RAS",
        )
        self.allen_ccf_space = AllenCCFv3Space()  # standard Allen CCF v3 space (PIR+ orientation)
        self.atlas = MRITorontoAtlas(res_um=10)  # The MRI Toronto brain atlas
        super().__init__(one=one, session=session)

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
        return {
            "exact_files_options": {
                "standard": [
                    f"alf/{FOV_name}/mpciMeanImage.mlapdv_estimate.npy",
                    f"alf/{FOV_name}/mpciMeanImage.brainLocationIds_ccf_2017_estimate.npy",
                ]
            },
        }

    # WARNING: The following method cannot be staticmethod due to self.FOV_name usage
    def get_load_object_kwargs(self) -> dict:
        """Return kwargs for one.load_object() call."""
        return {"obj": "mpciMeanImage", "collection": f"alf/{self.FOV_name}"}

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

    @classmethod
    def download_data(
        cls,
        one: ONE,
        eid: str,
        FOV_name: str,
        download_only: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> dict:
        """
        Download anatomical localization data.

        NOTE: Uses class-level REVISION attribute automatically.

        Parameters
        ----------
        one : ONE
            ONE API instance
        eid : str
            Session ID
        FOV_name : str
            Field of view name (e.g. "FOV_00", "FOV_01", etc.) to specify which ROI localization data to download
        download_only : bool, default=True
            If True, download but don't load into memory
        verbose : bool, default=False
            If True, print download status and timing information

        Returns
        -------
        dict
            Download status
        """
        requirements = cls.get_data_requirements(
            FOV_name=FOV_name
        )  # pass FOV_name from kwargs to get_data_requirements

        # Use class-level REVISION attribute
        revision = cls.REVISION

        start_time = time.time()

        # NO try-except - let it fail if files missing
        one.load_object(
            eid, obj="mpciMeanImage", collection=f"alf/{FOV_name}", download_only=download_only, revision=revision
        )

        download_time = time.time() - start_time

        if verbose:
            print(f"  Downloaded mean image data for {FOV_name} in {download_time:.2f}s")

        # SessionLoader handles format detection internally, report BWM format as default
        # (it will fall back to legacy if needed)
        return {
            "success": True,
            "downloaded_objects": ["mpciMeanImage"],
            "downloaded_files": requirements["exact_files_options"]["standard"],
            "already_cached": [],
            "alternative_used": None,
            "data": None,
        }

    def _add_coordinate_spaces(self, nwbfile: NWBFile):
        """Add coordinate spaces to the NWB file.

        Creates Space objects for the IBL bregma-centered coordinate system and the Allen CCF space,
        and adds them to a Localization container in the NWB file.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to which the coordinate spaces will be added.
        """

        if "localization" not in nwbfile.lab_meta_data:  # create Localization container if missing
            nwbfile.add_lab_meta_data([Localization()])

        localization = nwbfile.lab_meta_data["localization"]
        if (
            self.ibl_bregma_space.name not in localization.spaces
            and self.allen_ccf_space.name not in localization.spaces
        ):
            localization.add_spaces([self.ibl_bregma_space, self.allen_ccf_space])  # register both coordinate spaces

    def _ensure_mean_projection_image_exists(self, nwbfile: NWBFile):
        """Retrieve the mean projection image for this FOV from the NWB file.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to search.

        Returns
        -------
        GrayscaleImage
            The mean projection image for this FOV.

        Raises
        ------
        ValueError
            If the 'ophys' processing module, SegmentationImages container, or
            the mean image for this FOV does not exist.
        """
        if "ophys" not in nwbfile.processing:
            raise ValueError("No 'ophys' processing module found in NWB file.")

        summary_images_module = None
        for name, proc in nwbfile.processing["ophys"].data_interfaces.items():
            if name == "SegmentationImages":
                summary_images_module = nwbfile.processing["ophys"][name]
                break

        if summary_images_module is None:
            raise ValueError("No SegmentationImages data interface found in 'ophys' processing module.")

        mean_image = None
        if summary_images_module is not None:
            for mi_name, mi_object in summary_images_module.images.items():
                if self.camel_case_FOV_name in mi_name:
                    mean_image = mi_object
                    break
        if mean_image is None:
            raise ValueError(
                f"The mean image for {self.FOV_name} doesn't exist. "
                "Populate the SegmentationImages first "
                "(e.g. via MesoscopeSegmentationInterface in the processed pipeline) "
                "before running the anatomical localization interface."
            )
        return mean_image

    def _ensure_imaging_plane_exists(self, nwbfile: NWBFile):
        """Retrieve the ImagingPlane object for this FOV from the NWB file.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to search.

        Returns
        -------
        ImagingPalne
            The imaging plane for this FOV.

        Raises
        ------
        ValueError
            If ImagingPlane for this FOV does not exist.
        """
        if f"ImagingPlane{self.camel_case_FOV_name}" not in nwbfile.imaging_planes:
            raise ValueError(
                f"The imaging_plane for {self.FOV_name} doesn't exist. "
                f"Populate the ImagingPlane{self.camel_case_FOV_name} first "
                "(e.g. via MesoscopeMotionCorrectedImagingInterface in the processed pipeline) "
                "before running the anatomical localization interface."
            )
        return nwbfile.imaging_planes[f"ImagingPlane{self.camel_case_FOV_name}"]

    def _build_anatomical_coordinates_image(self, nwbfile: NWBFile):
        """Build AnatomicalCoordinatesImage objects for IBL-Bregma and Allen CCF v3 spaces.

        Loads per-pixel coordinates from `mpciMeanImage.mlapdv_estimate` (shape H×W×3,
        ML/AP/DV in µm, IBL-Bregma RAS space) and brain region IDs from
        `mpciMeanImage.brainLocationIds_ccf_2017_estimate`, then converts IBL-Bregma
        pixel coordinates to Allen CCF v3 (AP, DV, ML in µm) using the MRI Toronto atlas.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file containing the target mean image and imaging plane.

        Returns
        -------
        tuple[AnatomicalCoordinatesImage, AnatomicalCoordinatesImage]
            A pair of images: (IBL-Bregma image, Allen CCF v3 image).
            Each image stores per-pixel (x, y, z) coordinates matching the
            mean projection image dimensions (H×W).
        """
        mean_image = self._ensure_mean_projection_image_exists(nwbfile)
        imaging_plane = self._ensure_imaging_plane_exists(nwbfile)

        # Get mean image anatomical localization data
        mean_image_estimate = self.one.load_object(self.session, **self.get_load_object_kwargs())
        mean_image_mlapdv = mean_image_estimate["mlapdv_estimate"]
        mean_image_regions = mean_image_estimate["brainLocationIds_ccf_2017_estimate"]

        # Precompute brain region acronym lookup and build 2D acronym array
        unique_ids = np.unique(mean_image_regions)
        region_info = self.atlas.regions.get(unique_ids)
        id_to_acronym = dict(zip(unique_ids.tolist(), region_info.acronym.tolist()))
        brain_region_acronyms = np.vectorize(id_to_acronym.get)(mean_image_regions)

        # Create AnatomicalCoordinatesImage for IBL Bregma coordinates
        ibl_anatomical_coordinates_image = AnatomicalCoordinatesImage(
            name=f"AnatomicalCoordinatesImageIBLBregma{self.camel_case_FOV_name}",
            description=f"Mean image estimated coordinates in the IBL-Bregma coordinate system for {self.FOV_name}.",
            space=nwbfile.lab_meta_data["localization"].spaces[self.ibl_bregma_space.name],
            method="<>",  # TODO Add method description
            image=mean_image,
            x=mean_image_mlapdv[:, :, 0],
            y=mean_image_mlapdv[:, :, 1],
            z=mean_image_mlapdv[:, :, 2],
            brain_region=brain_region_acronyms,
            localized_entity=imaging_plane,
        )

        xyz_m_for_ccf = mean_image_mlapdv.reshape(-1, 3) / 1e6  # shape (H*W, 3), converted from um to m
        ccf_um = self.atlas.xyz2ccf(xyz=xyz_m_for_ccf, ccf_order="apdvml", mode="clip").astype(
            np.float64
        )  # shape (H, W, 3)
        mean_image_ccf = ccf_um.reshape(mean_image_mlapdv.shape)

        # Create AnatomicalCoordinatesImage for CCF coordinates
        ccf_anatomical_coordinates_image = AnatomicalCoordinatesImage(
            name=f"AnatomicalCoordinatesImageCCFv3{self.camel_case_FOV_name}",
            description=f"Mean image estimated coordinates in the CCF coordinate system for {self.FOV_name}.",
            space=nwbfile.lab_meta_data["localization"].spaces[self.allen_ccf_space.name],
            method="<>",  # TODO Add method description
            image=mean_image,
            x=mean_image_ccf[:, :, 0],
            y=mean_image_ccf[:, :, 1],
            z=mean_image_ccf[:, :, 2],
            brain_region=brain_region_acronyms,
            localized_entity=imaging_plane,
        )
        return ibl_anatomical_coordinates_image, ccf_anatomical_coordinates_image

    def _build_brain_region_masks(self):
        pass

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: Optional[dict] = None):
        """
        Add anatomical localization data to the NWB file.

        This method adds AnatomicalCoordinatesImage objects linking Mean Projection
        to IBL-Bregma and Allen CCF coordinate systems. The summary image container must already
        exist (done by MesoscopeSegmentationInterface).

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to add data to
        metadata : dict, optional
            Metadata dictionary (not currently used)

        Raises
        ------
        ValueError
            If plane segmentation table doesn't exist or is missing required columns
        """

        self._add_coordinate_spaces(nwbfile)
        localization = nwbfile.lab_meta_data["localization"]

        # Build AnatomicalCoordinatesImage
        ibl_anatomical_coordinates_image, ccf_anatomical_coordinates_image = self._build_anatomical_coordinates_image(
            nwbfile=nwbfile,
        )
        localization.add_anatomical_coordinates_images(
            [ibl_anatomical_coordinates_image, ccf_anatomical_coordinates_image]
        )

        # Build BrainRegionMasks for registered and source spaces
        # brain_region_masks = self._build_brain_region_masks()
        # localization.add_brain_region_masks(brain_region_masks)
