"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import datetime
import json
from pathlib import Path
from typing import Union
from zoneinfo import ZoneInfo

from natsort import natsorted
from neuroconv.utils import dict_deep_update, load_dict_from_file

from ibl_mesoscope_to_nwb.mesoscope2025 import RawMesoscopeNWBConverter
from ibl_mesoscope_to_nwb.mesoscope2025.metadata.update_mesoscope_ophys_metadata import (
    update_mesoscope_ophys_metadata,
)


def raw_session_to_nwb(
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    subject_id: str,
    eid: str,
    stub_test: bool = False,
    overwrite: bool = False,
):

    data_dir_path = Path(data_dir_path)
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    nwbfile_path = output_dir_path / f"sub-{subject_id}_ses-{eid}.nwb"

    source_data = dict()
    conversion_options = dict()

    # Add raw imaging data
    raw_imaging_folder = data_dir_path / "raw_imaging_data_00"
    tiff_files = natsorted(raw_imaging_folder.glob(f"*{subject_id}*.tif"))
    raw_imaging_metadata_path = (
        raw_imaging_folder / "_ibl_rawImagingData.meta.json"
    )  # TODO Confirm that metadata does not change from raw_imaging_data_00 and raw_imaging_data_01
    with open(raw_imaging_metadata_path, "r") as f:
        raw_metadata = json.load(f)
    num_planes = len(raw_metadata["FOV"])
    FOV_names = [f"FOV_{i:02d}" for i in range(num_planes)]
    for plane_index, FOV_name in enumerate(FOV_names):  # Limiting to first 2 FOVs for testing
        source_data.update(
            {
                f"{FOV_name}RawImaging": dict(
                    file_paths=tiff_files, plane_index=plane_index, channel_name="Channel 1", FOV_name=FOV_name
                )
            }
        )
        conversion_options.update({f"{FOV_name}RawImaging": dict(stub_test=stub_test, photon_series_index=plane_index)})

    converter = RawMesoscopeNWBConverter(source_data=source_data)

    # Add datetime to conversion
    metadata = converter.get_metadata()
    date = datetime.datetime(year=2020, month=1, day=1, tzinfo=ZoneInfo("US/Eastern"))
    metadata["NWBFile"]["session_start_time"] = date

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent.parent / "metadata" / "general_metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    # # Update ophys metadata
    # ophys_metadata_path = Path(__file__).parent / "metadata" / "mesoscope_ophys_metadata.yaml"
    # updated_ophys_metadata = update_mesoscope_ophys_metadata(
    #     ophys_metadata_path=ophys_metadata_path,
    #     raw_imaging_metadata_path=raw_imaging_metadata_path,
    #     FOV_names=FOV_names,
    # )
    # metadata = dict_deep_update(metadata, updated_ophys_metadata)

    metadata["Subject"]["subject_id"] = subject_id

    # Run conversion
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        overwrite=overwrite,
    )
