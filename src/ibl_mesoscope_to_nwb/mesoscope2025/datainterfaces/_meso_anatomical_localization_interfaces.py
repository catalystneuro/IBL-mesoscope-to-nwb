from pathlib import Path
from typing import Optional

import numpy as np
from ndx_anatomical_localization import (
    AnatomicalCoordinatesImage,
    AnatomicalCoordinatesTable,
    Localization,
    Space,
    AllenCCFv3Space,
)

from pydantic import DirectoryPath
from pynwb import NWBFile
from pynwb.image import Images
from pynwb.ophys import ImageSegmentation
from ibl_to_nwb.datainterfaces._base_ibl_interface import BaseIBLDataInterface


class MesoscopeAnatomicalLocalizationInterface(BaseIBLDataInterface):
    """A segmentation extractor for IBL Mesoscope."""

    interface_name = "MesoscopeAnatomicalLocalizationInterface"

    def __init__(
        self,
        folder_path: DirectoryPath,
        FOV_name: str | None = None,
    ):
        """Create SegmentationExtractor object out of suite 2p data type.

        Parameters
        ----------
        folder_path: str or Path
            The path to the 'alf' folder, where processed imaging data is stored.
        FOV_name: str, optional
            The name of the plane to load.
        """

        self.folder_path = Path(folder_path)
        self.FOV_name = FOV_name

        super().__init__()

        # Anatomical localization
        self._ROI_mlapdv_estimates_file_name = "mpciROIs.mlapdv_estimate.npy"
        self._ROI_brain_location_ids_estimates_file_name = "mpciROIs.brainLocationIds_ccf_2017_estimate.npy"
        self._mean_image_mlapdv_estimates_file_name = "mpciMeanImage.mlapdv_estimate.npy"
        self._mean_image_brain_location_ids_estimates_file_name = "mpciMeanImage.brainLocationIds_ccf_2017_estimate.npy"

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
                    f"alf/{FOV_name}/mpciMeanImage.mlapdv_estimate.npy",
                    f"alf/{FOV_name}/mpciMeanImage.brainLocationIds_ccf_2017_estimate.npy",
                ]
            },
        }

    # WARNING: The following method cannot be staticmethod due to self.task usage
    def get_load_object_kwargs(self) -> dict:
        """Return kwargs for one.load_object() call."""
        return {"obj": "wheel", "collection": f"alf/{self.task}"}

    @classmethod
    def check_availability(cls, one: ONE, eid: str, logger: Optional[logging.Logger] = None, **kwargs) -> dict:
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
        logger : logging.Logger, optional
            Logger for progress/warning messages
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
        quality_result = cls.check_quality(one=one, eid=eid, logger=logger, **kwargs)

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

    def _load_npy(self, file_name: str, mmap_mode=None, transpose: bool = False, require: bool = False):
        """Load a .npy file with specified filename. Returns None if file is missing.

        Parameters
        ----------
        file_name: str
            The name of the .npy file to load.
        mmap_mode: str
            The mode to use for memory mapping. See numpy.load for details.
        transpose: bool, optional
            Whether to transpose the loaded array.
        require: bool, optional
            Whether to raise an error if the file is missing.

        Returns
        -------
            The loaded .npy file.
        """
        file_path = self.folder_path / self.FOV_name / file_name
        if not file_path.exists():
            if require:
                raise FileNotFoundError(f"File {file_path} not found.")
            return

        data = np.load(file_path, mmap_mode=mmap_mode, allow_pickle=mmap_mode is None)
        if transpose:
            return data.T

        return data

    def get_rois_anatomical_localization(self, roi_ids=None) -> np.ndarray:
        """Get anatomical localization information for ROIs.

        Parameters
        ----------
        roi_ids : array-like, optional
            List of ROI IDs to get the anatomical localization for. If None, all ROIs are returned.
        Returns
        -------
        anatomical_localization: np.ndarray
            Array of anatomical localization information for the specified ROIs.
            Dimensions are (num_rois, 3) representing (ML, AP, DV) coordinates.
        """
        anatomical_localization = self._load_npy(
            file_name=self._ROI_mlapdv_estimates_file_name, require=True, transpose=False
        )
        assert (
            anatomical_localization is not None
        ), f"{self._ROI_mlapdv_estimates_file_name} is required but could not be loaded"
        return anatomical_localization if roi_ids is None else anatomical_localization[roi_ids, :]

    def get_rois_brain_location_ids(self, roi_ids=None) -> np.ndarray:
        """Get brain location IDs for ROIs.

        Parameters
        ----------
        roi_ids : array-like, optional
            List of ROI IDs to get the brain location IDs for. If None, all ROIs are returned.
        Returns
        -------
        brain_location_ids: np.ndarray
            Array of brain location IDs for the specified ROIs.
            Dimensions are (num_rois,) representing the brain location IDs.
        """
        brain_location_ids = self._load_npy(
            file_name=self._ROI_brain_location_ids_estimates_file_name, require=True, transpose=True
        )
        assert (
            brain_location_ids is not None
        ), f"{self._ROI_brain_location_ids_estimates_file_name} is required but could not be loaded"
        return brain_location_ids if roi_ids is None else brain_location_ids[roi_ids]

    def get_mean_image_anatomical_localization(self) -> np.ndarray:
        """Get anatomical localization information for the mean image.

        Returns
        -------
        anatomical_localization: np.ndarray
            Array of anatomical localization information for the mean image.
            Dimensions are (width, height, 3) representing (ML, AP, DV) coordinates for each pixel.
        """
        anatomical_localization = self._load_npy(file_name=self._mean_image_mlapdv_estimates_file_name, require=True)
        assert (
            anatomical_localization is not None
        ), f"{self._mean_image_mlapdv_estimates_file_name} is required but could not be loaded"
        return anatomical_localization

    def get_mean_image_brain_location_id(self) -> np.ndarray:
        """Get brain location ID for the mean image.

        Returns
        -------
        brain_location_id: np.ndarray
            Brain location ID for the mean image.
            Dimensions are (width, height) representing the brain location IDs for each pixel.
        """
        brain_location_id = self._load_npy(
            file_name=self._mean_image_brain_location_ids_estimates_file_name, require=True
        )
        assert (
            brain_location_id is not None
        ), f"{self._mean_image_brain_location_ids_estimates_file_name} is required but could not be loaded"
        return brain_location_id

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: Optional[dict] = None):
        """
        Add anatomical localization data to the NWB file.

        This method ONLY adds AnatomicalCoordinatesTable objects linking segmented ROIs
        to IBL-Bregma and CCF coordinate systems. The plane segmentation tables must already
        exist with anatomical columns populated (done by IblMesoscopeSegmentationInterface).

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

        # Create AnatomicalCoordinatesTable for CCF coordinates
        ibl_table = AnatomicalCoordinatesTable(
            name=f"ROIsIBLBregmaAnatomicalCoordinates{camel_case_FOV_name}",
            description=f"ROI centroid estimated coordinates in the IBL-Bregma coordinate system for {self.FOV_name}.",
            target=plane_segmentation,
            space=self.ibl_space,  # TODO: Verify this is correct
            method="TODO: Add method description",
        )
        ibl_table.add_column(
            name="brain_region_id",
            description="The brain region ID for the ROI in the plane segmentation table.",
        )

        # Get anatomical localization data
        rois_ccf_mlapdv = self.get_rois_anatomical_localization()
        rois_ccf_regions = self.get_rois_brain_location_ids()

        for roi_index in plane_segmentation.id[:]:
            ibl_table.add_row(
                localized_entity=roi_index,
                x=float(rois_ccf_mlapdv[roi_index][0]),
                y=float(rois_ccf_mlapdv[roi_index][1]),
                z=float(rois_ccf_mlapdv[roi_index][2]),
                brain_region_id=int(rois_ccf_regions[roi_index]),
                brain_region="TODO",
            )

        # Add tables to localization
        localization.add_anatomical_coordinates_tables([ibl_table])

        # Get mean image anatomical localization data
        mean_image_ibl_mlapdv = self.get_mean_image_anatomical_localization()
        mean_image_ibl_regions = self.get_mean_image_brain_location_id()

        ibl_image = AnatomicalCoordinatesImage(
            name=f"MeanImageIBLBregmaAnatomicalCoordinates{camel_case_FOV_name}",
            description=f"Mean image estimated coordinates in the IBL-Bregma coordinate system for {self.FOV_name}.",
            space=self.ibl_space,
            method="TODO: Add method description",
            image=mean_image,
            x=mean_image_ibl_mlapdv[:, :, 0],
            y=mean_image_ibl_mlapdv[:, :, 1],
            z=mean_image_ibl_mlapdv[:, :, 2],
            brain_region_id=mean_image_ibl_regions,
        )

        localization.add_anatomical_coordinates_images([ibl_image])

        # TODO: Add IBL-Bregma anatomical localization (check if the data are actually in Allen CCFv3 or IBL-Bregma)
