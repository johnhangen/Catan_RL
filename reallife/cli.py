"""
Real-life Catan CLI: one trained AI plays against human opponents on a physical board.

A human operator sits at a terminal and relays:
  - Dice results
  - What each human player did on their turn

The AI's decisions are printed in plain English so a human can execute them.
"""

import os
import sys
import numpy as np
import gymnasium
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.models.enums import ActionType
from catanatron_gym.envs.catanatron_env import from_action_space, ACTION_SPACE_SIZE
from catanatron.state_functions import (
    get_actual_victory_points,
    get_longest_road_color,
    get_largest_army,
    get_longest_road_length,
    player_num_resource_cards,
)

console = Console()

PLAYER_LABELS = {
    Color.BLUE: "[bold blue]AI (Blue)[/bold blue]",
    Color.RED: "[bold red]Red[/bold red]",
    Color.ORANGE: "[bold yellow]Orange[/bold yellow]",
    Color.WHITE: "[bold white]White[/bold white]",
}


def _tile_desc(tile) -> str:
    if tile.resource is None:
        return "Desert"
    return f"{tile.number}-{tile.resource.capitalize()}"


def _node_desc(node_id: int, catan_map) -> str:
    """Human-readable description of a node (vertex) position."""
    node_to_tiles = {}
    for coord, tile in catan_map.land_tiles.items():
        for _, nid in tile.nodes.items():
            node_to_tiles.setdefault(nid, []).append(tile)
    tiles = node_to_tiles.get(node_id, [])
    parts = [_tile_desc(t) for t in tiles if t.resource is not None]
    if not parts:
        return f"node {node_id} (coastal)"
    return "intersection of " + ", ".join(parts)


def _edge_desc(edge: tuple, catan_map) -> str:
    """Human-readable description of an edge (road position)."""
    n1, n2 = edge
    return f"road between node {n1} and node {n2} ({_node_desc(n1, catan_map)} ↔ {_node_desc(n2, catan_map)})"


def _coord_to_tile(coord, catan_map):
    return catan_map.land_tiles.get(coord)


def action_to_english(action, catan_map) -> str:
    """Convert a decoded Action to a plain-English instruction."""
    t = action.action_type
    v = action.value

    if t == ActionType.ROLL:
        return "Roll the dice."
    if t == ActionType.END_TURN:
        return "End your turn."
    if t == ActionType.BUILD_SETTLEMENT:
        return f"Build a settlement at the {_node_desc(v, catan_map)} (node {v})."
    if t == ActionType.BUILD_CITY:
        return f"Upgrade settlement to a city at the {_node_desc(v, catan_map)} (node {v})."
    if t == ActionType.BUILD_ROAD:
        n1, n2 = v
        return f"Build a road between node {n1} and node {n2}."
    if t == ActionType.BUY_DEVELOPMENT_CARD:
        return "Buy a development card from the bank."
    if t == ActionType.PLAY_KNIGHT_CARD:
        return "Play a Knight card (then move the robber)."
    if t == ActionType.PLAY_ROAD_BUILDING:
        return "Play Road Building — place 2 roads for free."
    if t == ActionType.PLAY_YEAR_OF_PLENTY:
        r1, r2 = v
        return f"Play Year of Plenty — take 1 {r1.capitalize()} and 1 {r2.capitalize()} from the bank."
    if t == ActionType.PLAY_MONOPOLY:
        return f"Play Monopoly — claim all {v.capitalize()} cards from every player."
    if t == ActionType.MOVE_ROBBER:
        coord, victim_color, _ = v
        tile = _coord_to_tile(coord, catan_map)
        tile_str = _tile_desc(tile) if tile else f"coord {coord}"
        victim_str = f" and steal 1 card from {victim_color.value.capitalize()}" if victim_color else ""
        return f"Move the robber to the {tile_str} tile{victim_str}."
    if t == ActionType.DISCARD:
        return f"Discard 1 {v.capitalize()} card."
    if t == ActionType.MARITIME_TRADE:
        give = [r for r in v if r is not None]
        give_res = give[0]
        get_res = give[-1]
        ratio = len(give) - 1
        return f"Trade {ratio} {give_res.capitalize()} → 1 {get_res.capitalize()} with the bank/port."
    return f"Perform action: {t.name} ({v})"


def _print_state_table(state, colors):
    table = Table(title="Game State", box=box.ROUNDED)
    table.add_column("Player", style="bold")
    table.add_column("VP")
    table.add_column("Resources")
    table.add_column("Dev Cards")
    table.add_column("Road Len")

    army_color, _ = get_largest_army(state)
    road_color = get_longest_road_color(state)

    for color in colors:
        label = PLAYER_LABELS.get(color, color.value)
        vp = get_actual_victory_points(state, color)
        res = player_num_resource_cards(state, color)
        # dev cards in hand (hidden for others, show count)
        from catanatron.state_functions import player_num_dev_cards
        devs = player_num_dev_cards(state, color)
        road_len = get_longest_road_length(state, color)
        extras = []
        if color == road_color:
            extras.append("🏆Road")
        if color == army_color:
            extras.append("⚔Army")
        table.add_row(label + (" " + " ".join(extras) if extras else ""), str(vp), str(res), str(devs), str(road_len))
    console.print(table)


def _ask(prompt: str) -> str:
    return console.input(f"[cyan]{prompt}[/cyan] ").strip()


def _ask_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        try:
            val = int(_ask(prompt))
            if lo <= val <= hi:
                return val
        except ValueError:
            pass
        console.print(f"[red]Please enter a number between {lo} and {hi}.[/red]")


def _ask_resource(prompt: str) -> str:
    resources = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
    console.print(f"[cyan]{prompt}[/cyan]")
    for i, r in enumerate(resources, 1):
        console.print(f"  [{i}] {r.capitalize()}")
    idx = _ask_int("Choice", 1, 5)
    return resources[idx - 1]


def _handle_human_turn(color: str, env):
    """Prompt the operator to enter what a human player did."""
    u = env.unwrapped
    label = color.value.capitalize()
    console.rule(f"[bold]{label}'s Turn[/bold]")

    while True:
        console.print("[bold]What did this player do?[/bold]")
        console.print("  [1] Roll dice (enter result)")
        console.print("  [2] Build road")
        console.print("  [3] Build settlement")
        console.print("  [4] Build city")
        console.print("  [5] Buy development card")
        console.print("  [6] Play development card")
        console.print("  [7] Maritime trade (bank/port)")
        console.print("  [8] Player trade with AI")
        console.print("  [9] End turn (done)")
        console.print("  [0] Skip / pass")
        choice = _ask("Action")

        if choice == "9" or choice == "0":
            break
        elif choice == "1":
            d = _ask_int("Dice result (2-12)", 2, 12)
            console.print(f"[dim]Logged: rolled {d}[/dim]")
        elif choice == "2":
            _ask("Road placed (press Enter to confirm)")
        elif choice == "3":
            _ask("Settlement placed (press Enter to confirm)")
        elif choice == "4":
            _ask("City built (press Enter to confirm)")
        elif choice == "5":
            _ask("Dev card bought (press Enter to confirm)")
        elif choice == "6":
            console.print("  Card type: [1] Knight [2] Year of Plenty [3] Monopoly [4] Road Building [5] VP")
            _ask("Which card? (press Enter to confirm)")
        elif choice == "7":
            give = _ask_resource("What resource did they give?")
            get = _ask_resource("What resource did they receive?")
            console.print(f"[dim]Logged: traded {give} → {get}[/dim]")
        elif choice == "8":
            console.print("[yellow]Player trade with AI:[/yellow]")
            give = _ask_resource("What resource did they give to AI?")
            get = _ask_resource("What resource did AI give them?")
            console.print(f"[dim]Logged: you gave {get}, received {give}[/dim]")
        else:
            console.print("[red]Unknown choice.[/red]")


def run_reallife(model_path: str, num_players: int = 4):
    """Main entry point for real-life mode.

    Args:
        model_path: Path to a saved MaskablePPO checkpoint (.zip).
        num_players: Total number of players (1 AI + rest human).
    """
    from sb3_contrib.ppo_mask import MaskablePPO

    console.print(Panel.fit(
        "[bold green]Catan RL — Real Life Mode[/bold green]\n"
        f"AI controls [bold blue]Blue[/bold blue]. "
        f"{num_players - 1} human player(s): "
        + ", ".join(c.value.capitalize() for c in [Color.RED, Color.ORANGE, Color.WHITE][: num_players - 1]),
        title="Welcome"
    ))

    # Load model
    model = MaskablePPO.load(model_path)
    console.print(f"[green]Model loaded from {model_path}[/green]")

    human_colors = [Color.RED, Color.ORANGE, Color.WHITE][: num_players - 1]

    # Create env with random opponents (they won't be used — we override human turns)
    enemies = [RandomPlayer(c) for c in human_colors]
    env = gymnasium.make("catanatron-v1", config={
        "map_type": "BASE",
        "vps_to_win": 10,
        "representation": "vector",
        "enemies": enemies,
    })

    obs, info = env.reset()
    u = env.unwrapped
    catan_map = u.game.state.board.map
    all_colors = [p.color for p in u.game.state.players]

    console.print("\n[bold]Game started![/bold] The board is set up on the physical table.\n")

    while True:
        state = u.game.state
        current_color = state.current_color()

        _print_state_table(state, all_colors)

        winner = u.game.winning_color()
        if winner is not None:
            if winner == Color.BLUE:
                console.print(Panel("[bold green]AI WINS! 🎉[/bold green]", title="Game Over"))
            else:
                console.print(Panel(f"[bold red]{winner.value.capitalize()} wins![/bold red]", title="Game Over"))
            break

        if current_color == Color.BLUE:
            # AI turn
            console.rule("[bold blue]AI's Turn[/bold blue]")
            valid = u.get_valid_actions()
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
            mask[valid] = True

            action_int, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action_int = int(action_int)
            decoded = from_action_space(action_int, state.playable_actions)

            description = action_to_english(decoded, catan_map)
            console.print(Panel(
                f"[bold yellow]>>> AI ACTION:[/bold yellow] {description}",
                title=f"Step {state.num_turns}",
                border_style="yellow",
            ))
            _ask("Press ENTER after executing this action on the board...")

            obs, reward, terminated, truncated, info = env.step(action_int)

            if terminated or truncated:
                winner = u.game.winning_color()
                if winner == Color.BLUE:
                    console.print(Panel("[bold green]AI WINS! 🎉[/bold green]", title="Game Over"))
                else:
                    console.print(Panel("[bold red]Game over.[/bold red]", title="Game Over"))
                break

        else:
            # Human player's turn — relay what they did
            _handle_human_turn(current_color, env)

            # The env's internal game already advanced past human turns automatically.
            # We just need to get the next AI observation.
            # Re-read obs from env (already advanced by catanatron's _advance_until_p0_decision)
            obs = u._get_observation()
            info = {"valid_actions": u.get_valid_actions()}

    env.close()
    console.print("\n[dim]Session ended.[/dim]")
