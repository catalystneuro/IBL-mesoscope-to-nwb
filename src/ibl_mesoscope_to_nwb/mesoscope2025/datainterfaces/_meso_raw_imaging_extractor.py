import numpy as np
from one.api import ONE
from roiextractors import MultiImagingExtractor, ScanImageImagingExtractor
from roiextractors.extraction_tools import PathType, get_package


class _SingleCollectionImagingExtractor(ScanImageImagingExtractor):
    """A segmentation extractor for reading IBL Raw Mesoscopic imaging data produced via ScanImage software in a single collection.

    This extractor is designed to handle the structure of ScanImage TIFF files, which can contain
    multi channel and both planar and volumetric data. It also supports both single-file and multi-file datasets generated
    by ScanImage in various acquisition modes (grab, focus, loop).

    The extractor creates a mapping between each frame in the dataset and its corresponding physical file
    and IFD (Image File Directory) location. This mapping enables efficient retrieval of specific frames
    without loading the entire dataset into memory, making it suitable for large datasets.

    The extractor also detects if ScanImage was using "Tiled" display mode during acquisition. In Tiled mode,
    multiple FOVs are stored vertically in a single TIFF frame with spacing between them.
    The extractor handles this configuration by extracting only the rows corresponding to the requested FOV, discarding the filler pixels.
    """

    extractor_name = "_SingleCollectionImagingExtractor"

    def __init__(
        self,
        FOV_index: int,
        file_path: PathType | None = None,
        channel_name: str | None = None,
        file_paths: list[PathType] | None = None,
    ):
        """
        Initialize the _SingleCollectionImagingExtractor.

        Parameters
        ----------
        file_path : PathType, optional
            Path to the IBL Mesoscope TIFF file. If this is part of a multi-file series, this should be the first file.
            Either `file_path` or `file_paths` must be provided.
        channel_name : str, optional
            Name of the channel to extract (e.g., "Channel 1", "Channel 2").
            - If None and only one channel is available, that channel will be used.
            - If None and multiple channels are available, an error will be raised.
            - Use `get_available_channel_names(file_path)` to see available channels before creating the extractor.
        file_paths : list[PathType], optional
            List of file paths to use. If provided, this overrides the automatic file detection heuristics.
            Use this parameter when:
            - Automatic detection doesn't work correctly
            - You need to specify a custom subset of files
            - You need to control the exact order of files
            The file paths must be provided in the temporal order of the frames in the dataset.
        FOV_index : int
            Index of the Field of View (FOV) to extract when using ScanImage "Tiled" display mode.

        Examples
        --------
        # Explicitly specifying multiple files
        >>> extractor = _SingleCollectionImagingExtractor(
        ...     file_paths=['path/to/file1.tif', 'path/to/file2.tif', 'path/to/file3.tif'],
        ...     channel_name='Channel 1'
        ... )
        """
        super().__init__(
            file_path=file_path,
            channel_name=channel_name,
            file_paths=file_paths,
            slice_sample=1,  # IBL acquisition for mesoscopic imaging is setup to acquire one frame per slice per channel
        )

        # Detect if ScanImage is using "Tiled" display mode
        # In Tiled mode, multiple FOVs are stored vertically in a single TIFF frame with spacing between them
        self.tiled_configuration = self._metadata["SI.hDisplay.volumeDisplayStyle"] == "Tiled"
        self.FOV_index = FOV_index

        if self.tiled_configuration:
            # TILED CONFIGURATION HANDLING:
            # When ScanImage uses Tiled display mode, each TIFF frame contains multiple FOVs
            # arranged vertically with filler pixels between them. This section handles the extraction
            # of a specific FOV from the composite frame.

            # Step 1: Determine the number of FOVs from the ScanImage ROI configuration
            # Each tile corresponds to one imaging ROI defined in the acquisition
            self.num_FOVs = len(self._general_metadata["RoiGroups"]["imagingRoiGroup"]["rois"])

            # Step 2: Read the TIFF file to get actual frame dimensions
            tifffile = get_package(package_name="tifffile")
            tiff_reader = tifffile.TiffReader(self.file_path)

            self._general_metadata = tiff_reader.scanimage_metadata
            # Get the actual dimensions of a single TIFF frame (contains all FOVs)
            sample_num_rows, sample_num_columns = tiff_reader.pages[0].shape

            # Step 3: Extract the dimensions of individual FOVs from ScanImage metadata
            # These represent the size of each FOV tile within the larger frame
            self._num_rows = self._metadata["SI.hRoiManager.linesPerFrame"]  # Rows per FOV (e.g., 512)
            self._num_columns = self._metadata["SI.hRoiManager.pixelsPerLine"]  # Columns per FOV (e.g., 512)

            # Step 4: Validate that tiles are distributed along rows (vertical stacking)
            # If columns don't match, it means tiles are arranged horizontally, which is not currently supported
            if sample_num_columns != self._num_columns:
                raise ValueError(
                    "Tiled configuration detected, but tiles are not distributed along rows. "
                    "This configuration is not yet supported."
                )

            # Step 5: Calculate the number of filler pixels between FOV tiles
            # Formula: (total_rows - rows_per_FOV × num_FOVs) / (num_FOVs - 1)
            # The filler pixels provide visual separation between tiles in the ScanImage display
            self._num_filler_pixels = (sample_num_rows - (self._num_rows * self.num_FOVs)) / (self.num_FOVs - 1)

            # Step 6: Validate that filler pixels are evenly distributed
            # The number must be an integer for consistent extraction
            if not self._num_filler_pixels.is_integer():
                raise ValueError(
                    "Tiled configuration detected, but the number of filler pixels between tiles is not consistent. "
                )

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
        """
        Get data as a time series from start_sample to end_sample.

        This method retrieves frames at the specified range from the ScanImage TIFF file(s).
        It uses the mapping created during initialization to efficiently locate and load only
        the requested frames, without loading the entire dataset into memory.

        For volumetric data (multiple planes), the returned array will have an additional dimension
        for the planes. For planar data (single plane), the plane dimension is squeezed out.

        Parameters
        ----------
        start_sample : int
        end_sample : int

        Returns
        -------
        numpy.ndarray
            Array of data with shape (num_samples, height, width) if num_planes is 1,
            or (num_samples, height, width, num_planes) if num_planes > 1.

            For example, for a non-volumetric dataset with 512x512 frames, requesting 3 samples
            would return an array with shape (3, 512, 512).

            For a volumetric dataset with 5 planes and 512x512 frames, requesting 3 samples
            would return an array with shape (3, 512, 512, 5).
        """
        start_sample = int(start_sample) if start_sample is not None else 0
        end_sample = int(end_sample) if end_sample is not None else self.get_num_samples()

        samples_in_series = end_sample - start_sample

        # Preallocate output array as volumetric and squeeze if not volumetric before returning
        num_rows, num_columns, num_planes = self.get_volume_shape()
        dtype = self.get_dtype()
        samples = np.empty((samples_in_series, num_rows, num_columns, num_planes), dtype=dtype)

        for return_index, sample_index in enumerate(range(start_sample, end_sample)):
            for depth_position in range(num_planes):

                # Calculate the index in the mapping table array
                frame_index = sample_index * num_planes + depth_position
                table_row = self._frames_to_ifd_table[frame_index]
                file_index = table_row["file_index"]
                ifd_index = table_row["IFD_index"]

                tiff_reader = self._tiff_readers[file_index]
                image_file_directory = tiff_reader.pages[ifd_index]
                if self.tiled_configuration:
                    # TILED FRAME EXTRACTION:
                    # In Tiled mode, each TIFF frame contains all FOVs stacked vertically.
                    # We need to extract only the rows corresponding to the requested FOV (FOV_index).
                    #
                    # Frame structure:
                    # FOV_0: rows [0, num_rows)
                    # Filler: rows [num_rows, num_rows + filler_pixels)
                    # FOV_1: rows [num_rows + filler_pixels, 2*num_rows + filler_pixels)
                    # Filler: rows [2*num_rows + filler_pixels, 2*num_rows + 2*filler_pixels)
                    # FOV_2: ...
                    #
                    # General formula for FOV_i:
                    # start_row = i × (num_rows + filler_pixels)
                    # end_row = start_row + num_rows
                    start_row = int(self.FOV_index * (self._num_rows + self._num_filler_pixels))
                    end_row = int(start_row + self._num_rows)
                    # Extract only the rows for the requested FOV, discarding filler pixels
                    samples[return_index, :, :, depth_position] = image_file_directory.asarray()[start_row:end_row, :]
                else:
                    # Non-tiled mode: the entire frame is the FOV
                    samples[return_index, :, :, depth_position] = image_file_directory.asarray()

        # Squeeze the depth dimension if not volumetric
        if not self.is_volumetric:
            samples = samples.squeeze(axis=3)

        return samples


class MesoscopeRawImagingExtractor(MultiImagingExtractor):
    """A multi session imaging extractor for reading IBL Raw Mesoscopic imaging data produced via ScanImage software.

    This extractor detect and consolidates multiple .tif files from multiple recording sessions (collections)
    into a single continuous dataset.
    It creates a MultiImagingExtractor by instantiating a _SingleCollectionImagingExtractor for each recording session and combining them.
    """

    extractor_name = "MesoscopeRawImagingExtractor"

    def __init__(
        self,
        one: ONE,
        session: str,
        FOV_index: int,
        channel_name: str,
        verbose: bool = False,
    ):
        local_session_folder_path = one.eid2path(session, query_type="local")
        raw_imaging_collections = one.list_collections(
            session,
            filename=f"*raw_imaging_data_*",
        )

        imaging_extractors = []

        for collection in raw_imaging_collections:
            if "reference" in collection:  # TODO: create a separate interface for reference imaging data
                continue
            else:
                decompressed_raw_imaging_folder = local_session_folder_path / collection / "imaging.frames"
                # check that folder exists
                if not decompressed_raw_imaging_folder.exists():
                    raise FileNotFoundError(
                        f"Decompressed raw imaging data folder not found at {decompressed_raw_imaging_folder}. Please ensure the raw imaging data has been decompressed."
                    )
                file_paths = sorted(decompressed_raw_imaging_folder.glob("*.tif"))

            if len(file_paths) == 0:
                raise FileNotFoundError(f"No .tif files found in collection {collection} for session {session}.")

            extractor = _SingleCollectionImagingExtractor(
                channel_name=channel_name, file_paths=file_paths, FOV_index=FOV_index
            )
            imaging_extractors.append(extractor)
            # imaging_extractor.set_times(times) #TODO: set correct timestamps for each frame based on IBL time alignment code

        super().__init__(imaging_extractors=imaging_extractors)
