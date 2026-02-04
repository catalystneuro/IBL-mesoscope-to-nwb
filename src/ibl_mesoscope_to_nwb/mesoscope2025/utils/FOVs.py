from one.api import ONE


def get_FOV_names_from_alf_collections(one: ONE, session: str) -> list[str]:
    """Get available tasks for a given session."""

    collections = one.list_collections(
        eid=session,
        filename="*alf/FOV*",
    )
    # sort by increasing FOV number
    return sorted([collection.split("/")[1] for collection in collections])


def get_number_of_FOVs_from_raw_imaging_metadata(one: ONE, session: str) -> int:
    """Get number of available FOVs for a given session. It is assum that task 00 exists and the number of FOVs is consistent across tasks."""
    metadata = one.load_dataset(session, dataset="_ibl_rawImagingData.meta.json", collection="raw_imaging_data_00")
    return len(metadata["FOV"])


if __name__ == "__main__":
    one = ONE()
    session = "5ce2e17e-8471-42d4-8a16-21949710b328"

    FOV_names = get_FOV_names_from_alf_collections(one, session)
    print(f"Available FOV names for session {session}: {FOV_names}")
