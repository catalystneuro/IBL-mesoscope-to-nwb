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
                output_path=output_path,
                append_on_disk_nwbfile=append_on_disk_nwbfile,
                verbose=verbose,
            )

        case "raw":
            return convert_raw_session(
                eid=eid,
                one=ONE(),  # base_url="https://alyx.internationalbrainlab.org"
                stub_test=stub_test,
                output_path=output_path,
                append_on_disk_nwbfile=append_on_disk_nwbfile,
                verbose=verbose,
            )
        case _:
            raise ValueError(f"Mode {mode} not recognized. Available modes: 'processed', 'raw'.")


if __name__ == "__main__":

    # Parameters for conversion
    output_path = Path("E:/IBL-data-share/IBL-mesoscope-nwbfiles")
    eid = "5ce2e17e-8471-42d4-8a16-21949710b328"
    stub_test = True  # Set to True for a quick test conversion with limited data
    start_time = time.time()
    mode = "processed"  # Choose between 'processed' and 'raw'
    session_to_nwb(
        output_path=output_path,
        eid=eid,
        mode=mode,
        stub_test=stub_test,
        append_on_disk_nwbfile=False,
        verbose=True,
    )
    print(f"Conversion completed in {time.time() - start_time:.2f} seconds.")
