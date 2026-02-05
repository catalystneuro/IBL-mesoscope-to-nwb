from typing import Optional

from ibl_to_nwb.datainterfaces._base_ibl_interface import BaseIBLDataInterface
from ndx_anatomical_localization import (
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
        self.one = one
        self.session = session
        self.revision = self.REVISION
        # Check if task exists
        FOV_names = get_FOV_names_from_alf_collections(one, session)
        if FOV_name not in FOV_names:
            raise ValueError(
                f"FOV_name '{FOV_name}' not found for session '{session}'. " f"Available FOV_names: {FOV_names}.'"
            )
        self.FOV_name = FOV_name

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
        camel_case_FOV_name = self.FOV_name.replace("_", "")
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
                if camel_case_FOV_name in ps_name:
                    plane_segmentation = ps_object
                    break
        if plane_segmentation is None:
            raise ValueError(
                f"Plane segmentation for {self.FOV_name} doesn't exist. "
                "Populate the plane segmentation table first "
                "(e.g. via IblMesoscopeSegmentationInterface in the processed pipeline) "
                "before running the anatomical localization interface."
            )
        if len(plane_segmentation) == 0:
            raise ValueError(
                f"Plane segmentation for {self.FOV_name} is empty. "
                "Populate the plane segmentation table first "
                "(e.g. via IblMesoscopeSegmentationInterface in the processed pipeline) "
                "before running the anatomical localization interface."
            )

        # Create or get the Localization container using dict.get
        localization = nwbfile.lab_meta_data.get("localization")
        if localization is None:
            localization = Localization()
            nwbfile.add_lab_meta_data(localization)

        # Create coordinate space objects
        ibl_space_name = "IBLBregma"
        if ibl_space_name not in localization.spaces:
            self.ibl_space = Space(
                name="IBLBregma",
                space_name="IBLBregma",
                origin="bregma",
                units="um",
                orientation="RAS",
            )
            localization.add_spaces(spaces=[self.ibl_space])
        else:
            self.ibl_space = localization.spaces[ibl_space_name]

        # Create AnatomicalCoordinatesTable for CCF coordinates
        ibl_table = AnatomicalCoordinatesTable(
            name=f"ROIsIBLBregmaAnatomicalCoordinates{camel_case_FOV_name}",
            description=f"ROI centroid estimated coordinates in the IBL-Bregma coordinate system for {self.FOV_name}.",
            target=plane_segmentation,
            space=self.ibl_space,
            method="TODO: Add method description",
        )
        ibl_table.add_column(
            name="brain_region_id",
            description="The brain region IDs are from the 2017 Allen CCF atlas.",
        )

        # Get anatomical localization data
        rois = self.one.load_object(self.session, **self.get_load_object_kwargs())
        rois_mlapdv = rois["mlapdv_estimate"]
        rois_brain_location_ids = rois["brainLocationIds_ccf_2017_estimate"]

        for roi_index in plane_segmentation.id[:]:
            ibl_table.add_row(
                localized_entity=roi_index,
                x=float(rois_mlapdv[roi_index][0]),
                y=float(rois_mlapdv[roi_index][1]),
                z=float(rois_mlapdv[roi_index][2]),
                brain_region_id=int(rois_brain_location_ids[roi_index]),
                #  brain_region= "TODO", add function that retrieves brain region name from id
            )

        # Add tables to localization
        localization.add_anatomical_coordinates_tables([ibl_table])


class MesoscopeImageAnatomicalLocalizationInterface(BaseIBLDataInterface):
    """An anatomical localization interface for Mean Projection."""

    interface_name = "MesoscopeImageAnatomicalLocalizationInterface"

    REVISION: str | None = None

    def __init__(self, one: ONE, session: str, FOV_name: str):
        self.one = one
        self.session = session
        self.revision = self.REVISION
        # Check if task exists
        FOV_names = get_FOV_names_from_alf_collections(one, session)
        if FOV_name not in FOV_names:
            raise ValueError(
                f"FOV_name '{FOV_name}' not found for session '{session}'. " f"Available FOV_names: {FOV_names}.'"
            )
        self.FOV_name = FOV_name

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

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: Optional[dict] = None):
        """
        Add anatomical localization data to the NWB file.

        This method ONLY adds AnatomicalCoordinatesImage objects linking Mean Projection
        to IBL-Bregma coordinate systems and CCF brain region IDS. The summary image container must already
        exist (done by IblMesoscopeSegmentationInterface).

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
        camel_case_FOV_name = self.FOV_name.replace("_", "")
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
                if camel_case_FOV_name in mi_name:
                    mean_image = mi_object
                    break
        if mean_image is None:
            raise ValueError(
                f"The mean image for {self.FOV_name} doesn't exist. "
                "Populate the SegmentationImages first "
                "(e.g. via IblMesoscopeSegmentationInterface in the processed pipeline) "
                "before running the anatomical localization interface."
            )
        # Create or get the Localization container using dict.get
        localization = nwbfile.lab_meta_data.get("localization")
        if localization is None:
            localization = Localization()
            nwbfile.add_lab_meta_data(localization)

        # Create coordinate space objects
        ibl_space_name = "IBLBregma"
        if ibl_space_name not in localization.spaces:
            self.ibl_space = Space(
                name="IBLBregma",
                space_name="IBLBregma",
                origin="bregma",
                units="um",
                orientation="RAS",
            )
            localization.add_spaces(spaces=[self.ibl_space])
        else:
            self.ibl_space = localization.spaces[ibl_space_name]

        # Get mean image anatomical localization data
        mean_image_estimate = self.one.load_object(self.session, **self.get_load_object_kwargs())
        mean_image_mlapdv = mean_image_estimate["mlapdv_estimate"]
        mean_image_regions = mean_image_estimate["brainLocationIds_ccf_2017_estimate"]

        ibl_image = AnatomicalCoordinatesImage(
            name=f"MeanImageIBLBregmaAnatomicalCoordinates{camel_case_FOV_name}",
            description=f"Mean image estimated coordinates in the IBL-Bregma coordinate system for {self.FOV_name}.",
            space=self.ibl_space,
            method="TODO: Add method description",
            image=mean_image,
            x=mean_image_mlapdv[:, :, 0],
            y=mean_image_mlapdv[:, :, 1],
            z=mean_image_mlapdv[:, :, 2],
            brain_region_id=mean_image_regions,
            # brain_region="TODO",  # add function that retrieves brain region name from id
        )

        localization.add_anatomical_coordinates_images([ibl_image])
