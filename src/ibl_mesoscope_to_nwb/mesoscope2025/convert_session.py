"""Primary script to run to convert an entire session for of data using the NWBConverter."""

from pathlib import Path
from typing import Union

from ibl_mesoscope_to_nwb.mesoscope2025.conversion.raw import raw_session_to_nwb


def session_to_nwb(
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    subject_id: str,
    eid: str,
    mode: str = "raw",
    stub_test: bool = False,
    overwrite: bool = False,
):
    match mode:
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
            raise ValueError(f"Mode {mode} not recognized. Available modes: 'raw'.")


if __name__ == "__main__":

    # Parameters for conversion
    data_dir_path = Path(r"E:\IBL-data-share\cortexlab\Subjects\SP061\2025-01-27\001")
    output_dir_path = Path(r"E:\ibl_mesoscope_conversion_nwb")
    eid = "42d7e11e-3185-4a79-a6ad-bbaf47366db2"
    stub_test = False  # Set to True for a quick test conversion with limited data

    session_to_nwb(
        data_dir_path=data_dir_path,
        output_dir_path=output_dir_path,
        subject_id="SP061",
        eid=eid,
        stub_test=stub_test,
        overwrite=True,
    )
