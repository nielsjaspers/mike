from nanobot.channels.telegram import TelegramChannel


def test_bot_commands_include_opencode_commands() -> None:
    commands = {command.command for command in TelegramChannel.BOT_COMMANDS}

    assert {"research", "status", "context"}.issubset(commands)


def test_bot_commands_still_include_core_defaults() -> None:
    commands = {command.command for command in TelegramChannel.BOT_COMMANDS}

    assert {"start", "new", "stop", "help", "restart"}.issubset(commands)
