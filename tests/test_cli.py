import pytest
from typer.testing import CliRunner

from arxiv_atlas.cli import app

COMMANDS = [
    "migrate",
    "status",
    "backfill",
    "update",
    "embed",
    "search",
    "fit-topics",
    "assign-topics",
    "topics",
]


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_parses_its_options(command):
    # Typer reads options from the function signatures; a bad annotation fails here.
    result = CliRunner().invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output
