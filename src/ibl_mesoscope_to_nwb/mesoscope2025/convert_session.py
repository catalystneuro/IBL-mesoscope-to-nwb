"""Primary script to run to convert an entire session for of data using the NWBConverter."""

from pathlib import Path
from typing import Union

from ibl_mesoscope_to_nwb.mesoscope2025.conversion import processed_session_to_nwb


def session_to_nwb(
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    subject_id: str,
    eid: str,
    mode: str = "processed",
    stub_test: bool = False,
    overwrite: bool = False,
):

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


if __name__ == "__main__":

    # Parameters for conversion
    data_dir_path = Path(r"E:\IBL-data-share\cortexlab\Subjects\SP061\2025-01-28\001")
    output_dir_path = Path(r"E:\ibl_mesoscope_conversion_nwb")
    eid = "5ce2e17e-8471-42d4-8a16-21949710b328"
    stub_test = True  # Set to True for a quick test conversion with limited data

    session_to_nwb(
        data_dir_path=data_dir_path,
        output_dir_path=output_dir_path,
        subject_id="SP061",
        eid=eid,
        stub_test=stub_test,
        overwrite=True,
    )
