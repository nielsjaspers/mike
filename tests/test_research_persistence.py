from nanobot.research.models import ResearchTask
from nanobot.research.persistence import ResearchPersistence


def test_persistence_saves_and_loads_task(tmp_path) -> None:
    store = ResearchPersistence(tmp_path)
    task = ResearchTask(
        task_id="abc123",
        session_key="telegram:1",
        origin_channel="telegram",
        origin_chat_id="1",
        query="query",
    )
    store.save_task(task)
    loaded = store.load_task("abc123")

    assert loaded is not None
    assert loaded.task_id == "abc123"


def test_persistence_writes_events_and_artifact(tmp_path) -> None:
    store = ResearchPersistence(tmp_path)
    store.append_event("abc123", "progress", {"phase": "PLANNING"})
    store.append_injection("abc123", "more context")
    artifact = store.write_artifact("abc123", "final.md", "hello")

    assert store.events_path("abc123").exists()
    assert store.injections_path("abc123").exists()
    assert artifact.endswith("final.md")
