"""Player whose actions are entered by a human operator at the terminal.

Used in real-life mode to mirror the actual moves a human opponent makes
on the physical board. Each time Catanatron asks this player to decide,
the operator picks from the legal action list.
"""

from rich.console import Console
from rich.table import Table
from rich import box
from catanatron.models.player import Player
from catanatron.models.enums import Action, ActionType

_console = Console()


def _prompt_dice():
    """Ask the operator for the physical dice result; returns a (d1, d2) tuple."""
    def _read_die(label):
        while True:
            raw = _console.input(f"[cyan]{label} (1-6):[/cyan] ").strip()
            try:
                v = int(raw)
                if 1 <= v <= 6:
                    return v
            except ValueError:
                pass
            _console.print("[red]Enter a number 1-6[/red]")
    return _read_die("Die 1"), _read_die("Die 2")

# Short labels for each action type when listed in the picker
_ACTION_TYPE_LABEL = {
    ActionType.ROLL: "Roll dice",
    ActionType.END_TURN: "End turn",
    ActionType.BUILD_ROAD: "Build road",
    ActionType.BUILD_SETTLEMENT: "Build settlement",
    ActionType.BUILD_CITY: "Build city",
    ActionType.BUY_DEVELOPMENT_CARD: "Buy dev card",
    ActionType.PLAY_KNIGHT_CARD: "Play Knight",
    ActionType.PLAY_ROAD_BUILDING: "Play Road Building",
    ActionType.PLAY_YEAR_OF_PLENTY: "Play Year of Plenty",
    ActionType.PLAY_MONOPOLY: "Play Monopoly",
    ActionType.MARITIME_TRADE: "Maritime trade",
    ActionType.MOVE_ROBBER: "Move robber",
    ActionType.DISCARD: "Discard",
}


def _format_action(action) -> str:
    label = _ACTION_TYPE_LABEL.get(action.action_type, action.action_type.name)
    v = action.value
    if v is None:
        return label
    return f"{label} {v}"


class HumanRelayPlayer(Player):
    """Opponent that prompts the human operator for each action."""

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        name = self.color.value.capitalize()

        # If there is only one legal action, take it automatically.
        # Special case: ROLL needs the real dice values injected into Action.value
        # so resource production matches the physical board.
        if len(actions) == 1:
            only = actions[0]
            if only.action_type == ActionType.ROLL:
                _console.print(f"[bold]{name} rolls the dice.[/bold]")
                d1, d2 = _prompt_dice()
                return Action(self.color, ActionType.ROLL, (d1, d2))
            _console.print(f"[dim]{name} auto-plays only legal action: {_format_action(only)}[/dim]")
            return only

        _console.rule(f"[bold]{name}'s Turn[/bold]")
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("Action")
        for i, a in enumerate(actions):
            table.add_row(str(i), _format_action(a))
        _console.print(table)

        while True:
            raw = _console.input(f"[cyan]What did {name} do? (number, or 'list' to reshow):[/cyan] ").strip()
            if raw.lower() == "list":
                _console.print(table)
                continue
            try:
                idx = int(raw)
                if 0 <= idx < len(actions):
                    chosen = actions[idx]
                    if chosen.action_type == ActionType.ROLL:
                        d1, d2 = _prompt_dice()
                        return Action(self.color, ActionType.ROLL, (d1, d2))
                    return chosen
            except ValueError:
                pass
            _console.print(f"[red]Enter a number 0-{len(actions) - 1}[/red]")
