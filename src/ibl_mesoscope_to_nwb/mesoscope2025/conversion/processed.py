"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import datetime
from pathlib import Path
from typing import Union
from zoneinfo import ZoneInfo

from neuroconv.utils import dict_deep_update, load_dict_from_file

from ibl_mesoscope_to_nwb.mesoscope2025 import ProcessedMesoscopeNWBConverter
from ibl_mesoscope_to_nwb.mesoscope2025.datainterfaces import (  # noqa: F401
    IBLMesoscopeMotionCorrectedImagingExtractor,
    IBLMesoscopeSegmentationExtractor,
)
from ibl_mesoscope_to_nwb.mesoscope2025.metadata.update_mesoscope_ophys_metadata import (
    update_mesoscope_ophys_metadata,
)


def processed_session_to_nwb(
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

    # Add Motion Corrected Imaging
    mc_imaging_folder = data_dir_path / "suite2p"
    available_planes = IBLMesoscopeMotionCorrectedImagingExtractor.get_available_planes(mc_imaging_folder)
    available_planes = available_planes[:2] if stub_test else available_planes  # Limit to first 2 planes for testing
    for plane_number, plane_name in enumerate(available_planes):
        file_path = mc_imaging_folder / plane_name / "imaging.frames_motionRegistered.bin"
        source_data.update({f"{plane_name}MotionCorrectedImaging": dict(file_path=file_path)})
        conversion_options.update(
            {f"{plane_name}MotionCorrectedImaging": dict(stub_test=False, photon_series_index=plane_number)}
        )

    # Add Segmentation
    segmentation_folder = data_dir_path / "alf"
    FOV_names = IBLMesoscopeSegmentationExtractor.get_available_planes(segmentation_folder)
    FOV_names = FOV_names[:2] if stub_test else FOV_names  # Limit to first 2 planes for testing
    for plane_name in FOV_names:
        source_data.update({f"{plane_name}Segmentation": dict(folder_path=segmentation_folder, plane_name=plane_name)})
        conversion_options.update({f"{plane_name}Segmentation": dict(stub_test=stub_test)})

    converter = ProcessedMesoscopeNWBConverter(source_data=source_data)

    # Add datetime to conversion
    metadata = converter.get_metadata()
    date = datetime.datetime(year=2020, month=1, day=1, tzinfo=ZoneInfo("US/Eastern"))
    metadata["NWBFile"]["session_start_time"] = date

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "metadata" / "general_metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    # Update ophys metadata
    ophys_metadata_path = Path(__file__).parent / "metadata" / "mesoscope_ophys_metadata.yaml"
    raw_imaging_metadata_path = (
        data_dir_path / "raw_imaging_data_00" / "_ibl_rawImagingData.meta.json"
    )  # TODO Confirm that metadata does not change from raw_imaging_data_00 and raw_imaging_data_01
    updated_ophys_metadata = update_mesoscope_ophys_metadata(
        ophys_metadata_path=ophys_metadata_path,
        raw_imaging_metadata_path=raw_imaging_metadata_path,
        FOV_names=FOV_names,
    )
    metadata = dict_deep_update(metadata, updated_ophys_metadata)

    metadata["Subject"]["subject_id"] = subject_id

    # Run conversion
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        overwrite=overwrite,
    )
