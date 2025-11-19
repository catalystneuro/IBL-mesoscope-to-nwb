"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import time
from pathlib import Path
from typing import Literal, Union

from ibl_mesoscope_to_nwb.mesoscope2025.conversion import (
    processed_session_to_nwb,
    raw_session_to_nwb,
)


def session_to_nwb(
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    subject_id: str,
    eid: str,
    mode: Literal["processed", "raw"],
    stub_test: bool = False,
    overwrite: bool = False,
):
    """
    Convert a single session to NWB format.

    Parameters
    ----------
    data_dir_path : Union[str, Path]
        Path to the directory containing the session data.
    output_dir_path : Union[str, Path]
        Path to the directory where the NWB file will be saved.
    subject_id : str
        The subject ID for the session.
    eid : str
        The experiment ID (session ID) for the session.
    mode : Literal["processed", "raw"]
        The conversion mode to use.
    stub_test : bool, optional
        Whether to run a stub test with limited data, by default False.
    overwrite : bool, optional
        Whether to overwrite existing NWB files, by default False.
    """
    match mode:
        case "processed":
            processed_session_to_nwb(
                data_dir_path=data_dir_path,
                output_dir_path=output_dir_path,
                subject_id=subject_id,
                eid=eid,
                stub_test=stub_test,
                overwrite=overwrite,
            )
        case "raw":
            return raw_session_to_nwb(
                data_dir_path=data_dir_path,
                output_dir_path=output_dir_path,
                subject_id=subject_id,
                eid=eid,
                stub_test=stub_test,
                overwrite=overwrite,
            )
        case _:
            raise ValueError(f"Mode {mode} not recognized. Available modes: 'processed', 'raw'.")


if __name__ == "__main__":

    # Parameters for conversion
    data_dir_path = Path(r"E:\IBL-data-share\cortexlab\Subjects\SP061\2025-01-27\001")
    output_dir_path = Path(r"E:\ibl_mesoscope_conversion_nwb")
    eid = "42d7e11e-3185-4a79-a6ad-bbaf47366db2"
    stub_test = False  # Set to True for a quick test conversion with limited data
    start_time = time.time()
    session_to_nwb(
        data_dir_path=data_dir_path,
        output_dir_path=output_dir_path,
        subject_id="SP061",
        eid=eid,
        mode="raw",
        stub_test=stub_test,
        overwrite=True,
    )
    print(f"Conversion completed in {time.time() - start_time:.2f} seconds.")
