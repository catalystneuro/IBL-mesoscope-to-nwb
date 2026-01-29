from one.api import ONE


def get_available_FOV_names(one: ONE, session: str) -> list[str]:
    """Get available tasks for a given session."""

    collections = one.list_collections(
        eid=session,
        filename="*alf/FOV*",
    )
    # sort by increasing FOV number
    return sorted([collection.split("/")[1] for collection in collections])


if __name__ == "__main__":
    one = ONE()
    session = "5ce2e17e-8471-42d4-8a16-21949710b328"

    FOV_names = get_available_FOV_names(one, session)
    print(f"Available FOV names for session {session}: {FOV_names}")
