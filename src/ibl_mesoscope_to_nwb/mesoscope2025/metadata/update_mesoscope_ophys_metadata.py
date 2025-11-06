from pathlib import Path
from neuroconv.utils import load_dict_from_file
from typing import List


def update_mesoscope_ophys_metadata(
    ophys_metadata_path: Path, raw_imaging_metadata_path: Path, FOV_names: List[str]
) -> dict:
    """
    1. Load the metadata structure from ophys metadata from a YAML file.
    2. Load actual values from `_ibl_rawImagingData.meta.json` file that contains comprehensive metadata about the mesoscopic imaging acquisition,
    including ScanImage configuration, ROI definitions, laser settings, and coordinate transformations.
    3. Iterate through each imaging plane defined in the raw imaging metadata.

    Parameters
    ----------
    ophys_metadata_path : Path
        Path to the YAML file containing the ophys metadata structure.
    raw_imaging_metadata_path : Path
        Path to the `_ibl_rawImagingData.meta.json` file containing actual metadata values.
    FOV_names : List[str]
        List of field of view (FOV) names to process. E.g. ['FOV_00', 'FOV_01', ...]

    Returns
    -------
    dict
        The updated metadata dictionary containing both the structure and the actual values for each FOV.
    """

    # Update ophys metadata
    ophys_metadata = load_dict_from_file(ophys_metadata_path)

    return ophys_metadata
