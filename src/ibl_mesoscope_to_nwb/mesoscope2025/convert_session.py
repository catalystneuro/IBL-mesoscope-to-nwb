"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import datetime
from pathlib import Path
from typing import Union
from zoneinfo import ZoneInfo

from neuroconv.utils import dict_deep_update, load_dict_from_file

from ibl_mesoscope_to_nwb.mesoscope2025 import Mesoscope2025NWBConverter
from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import (  # noqa: F401
    IBLMesoscopeSegmentationExtractor,
)


def session_to_nwb(
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    subject_id: str,
    session_id: str,
    stub_test: bool = False,
    overwrite: bool = False,
):

    data_dir_path = Path(data_dir_path)
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    nwbfile_path = output_dir_path / f"{session_id}.nwb"

    source_data = dict()
    conversion_options = dict()

    # Add Segmentation
    available_planes = IBLMesoscopeSegmentationExtractor.get_available_planes(data_dir_path)
    for plane_name in available_planes:
        source_data.update({f"{plane_name}Segmentation": dict(folder_path=data_dir_path, plane_name=plane_name)})
        conversion_options.update(
            {f"{plane_name}Segmentation": dict(stub_test=stub_test, iterator_option=dict(display_progress=True))}
        )
    converter = Mesoscope2025NWBConverter(source_data=source_data)

    # Add datetime to conversion
    metadata = converter.get_metadata()
    date = datetime.datetime(year=2020, month=1, day=1, tzinfo=ZoneInfo("US/Eastern"))
    metadata["NWBFile"]["session_start_time"] = date

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    metadata["Subject"]["subject_id"] = subject_id

    # Run conversion
    converter.run_conversion(
        metadata=metadata, nwbfile_path=nwbfile_path, conversion_options=conversion_options, overwrite=overwrite
    )


if __name__ == "__main__":

    # Parameters for conversion
    data_dir_path = Path(r"F:\IBL-data-share\cortexlab\Subjects\SP061\2025-01-28\001\alf")
    output_dir_path = Path(r"F:\ibl_mesoscope_conversion_nwb")
    stub_test = False

    session_to_nwb(
        data_dir_path=data_dir_path,
        output_dir_path=output_dir_path,
        subject_id="SP061",
        session_id="2025-01-28-001",
        stub_test=stub_test,
        overwrite=True,
    )
