"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import time
from pathlib import Path
from typing import Literal

from one.api import ONE

from ibl_mesoscope_to_nwb.mesoscope2025.conversion import (
    convert_processed_session,
    convert_raw_session,
    download_processed_session_data,
    download_raw_session_data,
)


def session_to_nwb(
    output_path: Path,
    eid: str,
    mode: Literal["processed", "raw"],
    stub_test: bool = False,
    append_on_disk_nwbfile: bool = False,
    verbose: bool = False,
):
    """
    Convert a single session to NWB format.

    Downloads all required data before conversion using each interface's
    ``download_data`` classmethod.

    Parameters
    ----------
    output_path : Path
        Base path to the directory where the NWB files will be saved.
    eid : str
        The experiment ID (session ID) for the session.
    mode : Literal["processed", "raw"]
        The conversion mode to use.
    stub_test : bool, optional
        Whether to run a stub test with limited data, by default False.
    append_on_disk_nwbfile : bool, optional
        Whether to append data to an existing on-disk NWB file, by default False.
    verbose : bool, optional
        Whether to print detailed conversion information, by default False.
    """
    one = ONE()

    match mode:
        case "processed":
            # ----------------------------------------------------------------
            # STEP 1: Download data
            # ----------------------------------------------------------------
            download_processed_session_data(eid=eid, one=one, stub_test=stub_test, verbose=verbose)

            # ----------------------------------------------------------------
            # STEP 2: Convert
            # ----------------------------------------------------------------
            return convert_processed_session(
                eid=eid,
                one=one,
                stub_test=stub_test,
                output_path=output_path,
                append_on_disk_nwbfile=append_on_disk_nwbfile,
                verbose=verbose,
            )

        case "raw":
            # ----------------------------------------------------------------
            # STEP 1: Download data
            # ----------------------------------------------------------------
            download_raw_session_data(eid=eid, one=one, stub_test=stub_test, verbose=verbose)

            # ----------------------------------------------------------------
            # STEP 2: Convert
            # ----------------------------------------------------------------
            return convert_raw_session(
                eid=eid,
                one=one,
                stub_test=stub_test,
                output_path=output_path,
                append_on_disk_nwbfile=append_on_disk_nwbfile,
                verbose=verbose,
            )
        case _:
            raise ValueError(f"Mode {mode} not recognized. Available modes: 'processed', 'raw'.")


if __name__ == "__main__":
    eids = [
        "5ce2e17e-8471-42d4-8a16-21949710b328",
    ]
    # Parameters for conversion
    output_path = Path("E:/IBL-mesoscope-nwbfiles")
    stub_test = False  # Set to True for a quick test conversion with limited data
    for eid in eids:
        start_time = time.time()
        try:
            session_to_nwb(
                output_path=output_path,
                eid=eid,
                mode="processed",
                stub_test=stub_test,
                append_on_disk_nwbfile=False,
                verbose=True,
            )
            print(f"Conversion completed in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"Conversion failed for eid {eid} with error: {e}")
        start_time = time.time()
        try:
            session_to_nwb(
                output_path=output_path,
                eid=eid,
                mode="raw",
                stub_test=stub_test,
                append_on_disk_nwbfile=False,
                verbose=True,
            )
            print(f"Conversion completed in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"Conversion failed for eid {eid} with error: {e}")
