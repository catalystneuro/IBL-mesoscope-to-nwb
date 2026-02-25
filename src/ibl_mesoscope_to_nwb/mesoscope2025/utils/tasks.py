from one.api import ONE


def get_available_tasks_from_alf_collections(one: ONE, session: str) -> list[str]:
    """Get available tasks for a given session."""

    collections = one.list_collections(
        eid=session,
        filename="*alf/task*",
    )
    return [collection.split("/")[1] for collection in collections]


if __name__ == "__main__":
    one = ONE()
    session = "5ce2e17e-8471-42d4-8a16-21949710b328"
    tasks = get_available_tasks_from_alf_collections(one, session)
    print(f"Available tasks for session {session}: {tasks}")
