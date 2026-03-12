from nanobot.session.manager import Session


def test_session_history_preserves_reasoning_content() -> None:
    session = Session(key="telegram:123")
    session.add_message(
        "assistant",
        "",
        tool_calls=[
            {"id": "abc", "type": "function", "function": {"name": "x", "arguments": "{}"}}
        ],
        reasoning_content="hidden reasoning",
    )

    history = session.get_history(max_messages=10)

    assert history[0]["reasoning_content"] == "hidden reasoning"
