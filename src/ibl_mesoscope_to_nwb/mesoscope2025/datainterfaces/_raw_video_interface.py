"""Data Interface for the raw behavioral video."""

import re
from pathlib import Path
from shutil import copyfile

import numpy as np
from neuroconv.basedatainterface import BaseDataInterface
from pydantic import DirectoryPath
from pynwb.image import ImageSeries


class RawVideoInterface(BaseDataInterface):
    """Interface for raw behavioral video data."""

    def __init__(
        self,
        folder_path: DirectoryPath,
        nwbfiles_folder_path: DirectoryPath,
        subject_id: str,
        session: str,
        camera_name: str,
    ):
        """Initialize the RawVideoInterface.
        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the folder containing the "raw_video_data" folder and "alf" folder.
        nwbfiles_folder_path : DirectoryPath
            The folder path where the NWB file will be written in DANDI organization structure.
            This is an unusual value to pass to __init__, but in this case it is necessary to simplify the DANDI
            organization of the externally stored raw video data.
        subject_id : str
            The subject ID to use for the DANDI organization. This is also an unusual value to pass to __init__, but
            the custom handling of Subject extensions requires removing it from the main metadata at runtime.
        session : str
            The session ID (EID in ONE).
        camera_name : str
            Name of the camera (e.g., 'camera_left', 'camera_right', 'camera_body').

        """

        self.folder_path = Path(folder_path)
        self.nwbfiles_folder_path = Path(nwbfiles_folder_path)

        super().__init__()
        self.camera_view = re.search(r"(left|right|body)", camera_name).group(1)
        self.video_filename = f"raw_video_data/_iblrig_{camera_name}.raw.mp4"
        self.camera_times_file_name = f"alf/_ibl_{camera_name}.times.npy"
        self.camera_name = camera_name
        self.subject_id = subject_id
        self.session = session

    def add_to_nwbfile(self, nwbfile, metadata: dict):
        camera_times = np.load(self.folder_path / self.camera_times_file_name, mmap_mode=None, allow_pickle=True)

        # Rename to DANDI format and relative organization
        dandi_sub_stem = f"sub-{self.subject_id}"
        dandi_subject_folder = self.nwbfiles_folder_path / dandi_sub_stem

        dandi_sub_ses_stem = f"{dandi_sub_stem}_ses-{self.session}"
        dandi_video_folder_path = dandi_subject_folder / f"{dandi_sub_ses_stem}_ecephys+image"
        dandi_video_folder_path.mkdir(exist_ok=True, parents=True)

        nwb_video_name = f"Video{self.camera_name.capitalize()}Camera"
        dandi_video_file_path = dandi_video_folder_path / f"{dandi_sub_ses_stem}_{nwb_video_name}.mp4"

        # A little bit of data duplication to copy, but easier for re-running since original file stays in cache
        original_video_file_path = self.folder_path / self.video_filename
        copyfile(src=original_video_file_path, dst=dandi_video_file_path)

        image_series = ImageSeries(
            name=nwb_video_name,
            description="Raw video from camera recording behavioral and task events.",
            unit="n.a.",
            external_file=["./" + str(dandi_video_file_path.relative_to(dandi_subject_folder))],
            format="external",
            timestamps=camera_times,
        )
        nwbfile.add_acquisition(image_series)

        # TODO add device camera with spec
