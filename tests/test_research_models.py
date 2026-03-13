from nanobot.research.models import ResearchTask


def test_research_task_round_trip() -> None:
    task = ResearchTask(
        task_id="abc123",
        session_key="telegram:1",
        origin_channel="telegram",
        origin_chat_id="1",
        query="What should I know about Hetzner + Pi 5 research agents?",
    )
    task.model = "minimax-m2.5"
    data = task.to_dict()
    restored = ResearchTask.from_dict(data)

    assert restored.task_id == task.task_id
    assert restored.model == "minimax-m2.5"


def test_research_task_tracks_progress() -> None:
    task = ResearchTask(
        task_id="abc123",
        session_key="telegram:1",
        origin_channel="telegram",
        origin_chat_id="1",
        query="query",
    )
    task.progress_summary = "still researching"
    data = task.to_dict()
    restored = ResearchTask.from_dict(data)

    assert restored.progress_summary == "still researching"
