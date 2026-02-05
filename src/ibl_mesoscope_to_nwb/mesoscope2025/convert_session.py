"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import time
from pathlib import Path
from typing import Literal

from one.api import ONE

from ibl_mesoscope_to_nwb.mesoscope2025.conversion import (
    convert_processed_session,
    convert_raw_session,
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
    match mode:
        case "processed":
            return convert_processed_session(
                eid=eid,
                one=ONE(),  # base_url="https://alyx.internationalbrainlab.org"
                stub_test=stub_test,
                output_path=output_path / "processed",
                append_on_disk_nwbfile=append_on_disk_nwbfile,
                verbose=verbose,
            )

        case "raw":
            return convert_raw_session(
                eid=eid,
                one=ONE(),  # base_url="https://alyx.internationalbrainlab.org"
                stub_test=stub_test,
                output_path=output_path / "raw",
                append_on_disk_nwbfile=append_on_disk_nwbfile,
                verbose=verbose,
            )
        case _:
            raise ValueError(f"Mode {mode} not recognized. Available modes: 'processed', 'raw'.")


if __name__ == "__main__":
    eids = [
        "5ce2e17e-8471-42d4-8a16-21949710b328",
        "42d7e11e-3185-4a79-a6ad-bbaf47366db2",
        "4693e7cc-17f6-4eeb-8abb-5951ba82b601",
        "e7c3df94-ef2a-44ed-a8e3-9d1a995b54f9",
        "c13eb6d3-09f5-49f7-bd89-26fce25ff65f",
        "1e558505-7d94-4851-83ef-edb2844ee805",
        "6f12a581-2203-4cd3-97b4-cd9cd78b440e",
    ]
    # Parameters for conversion
    output_path = Path("E:/IBL-mesoscope-nwbfiles")
    stub_test = True  # Set to True for a quick test conversion with limited data
    mode = "processed"  # Choose between 'processed' and 'raw'
    for eid in eids:
        start_time = time.time()
        try:
            session_to_nwb(
                output_path=output_path,
                eid=eid,
                mode=mode,
                stub_test=stub_test,
                append_on_disk_nwbfile=False,
                verbose=True,
            )
            print(f"Conversion completed in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"Conversion failed for eid {eid} with error: {e}")
