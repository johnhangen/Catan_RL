"""
Real-life Catan CLI: one trained AI plays against human opponents on a physical board.

The AI's chosen action is printed in plain English so the operator can execute it
on the real board. When opponent turns come up, a HumanRelayPlayer prompts the
operator to pick the action the human player actually took.
"""

import numpy as np
import gymnasium
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from catanatron.models.player import Color
from catanatron.models.enums import Action, ActionType
from catanatron_gym.envs.catanatron_env import from_action_space, ACTION_SPACE_SIZE
from catanatron.state_functions import (
    get_actual_victory_points,
    get_longest_road_color,
    get_largest_army,
    get_longest_road_length,
    player_num_resource_cards,
    player_num_dev_cards,
)
from agents.human_relay_player import HumanRelayPlayer, _prompt_dice

console = Console()

PLAYER_LABELS = {
    Color.BLUE: "[bold blue]AI (Blue)[/bold blue]",
    Color.RED: "[bold red]Red[/bold red]",
    Color.ORANGE: "[bold yellow]Orange[/bold yellow]",
    Color.WHITE: "[bold white]White[/bold white]",
}


def _tile_desc(tile) -> str:
    if tile is None or tile.resource is None:
        return "Desert"
    return f"{tile.number}-{tile.resource.capitalize()}"


def _build_node_to_tiles(catan_map):
    node_to_tiles = {}
    for _, tile in catan_map.land_tiles.items():
        for _, nid in tile.nodes.items():
            node_to_tiles.setdefault(nid, []).append(tile)
    return node_to_tiles


def _node_desc(node_id: int, node_to_tiles: dict) -> str:
    tiles = node_to_tiles.get(node_id, [])
    parts = [_tile_desc(t) for t in tiles if t.resource is not None]
    if not parts:
        return f"node {node_id} (coastal)"
    return "intersection of " + ", ".join(parts)


def action_to_english(action, catan_map, node_to_tiles) -> str:
    """Convert a decoded Action to a plain-English instruction for the operator."""
    t = action.action_type
    v = action.value

    if t == ActionType.ROLL:
        return "Roll the dice."
    if t == ActionType.END_TURN:
        return "End your turn."
    if t == ActionType.BUILD_SETTLEMENT:
        return f"Build a settlement at the {_node_desc(v, node_to_tiles)} (node {v})."
    if t == ActionType.BUILD_CITY:
        return f"Upgrade settlement to a city at the {_node_desc(v, node_to_tiles)} (node {v})."
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
        tile = catan_map.land_tiles.get(coord)
        tile_str = _tile_desc(tile) if tile else f"coord {coord}"
        victim_str = (
            f" and steal 1 card from {victim_color.value.capitalize()}"
            if victim_color else ""
        )
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
        devs = player_num_dev_cards(state, color)
        road_len = get_longest_road_length(state, color)
        extras = []
        if color == road_color:
            extras.append("Road")
        if color == army_color:
            extras.append("Army")
        suffix = f" [{' '.join(extras)}]" if extras else ""
        table.add_row(label + suffix, str(vp), str(res), str(devs), str(road_len))
    console.print(table)


def run_reallife(model_path: str, num_players: int = 4):
    """Main entry point for real-life mode.

    Args:
        model_path: Path to a saved MaskablePPO checkpoint (.zip).
        num_players: Total players (1 AI + the rest are real humans).
    """
    from sb3_contrib.ppo_mask import MaskablePPO

    if num_players < 2 or num_players > 4:
        raise ValueError("num_players must be 2-4")

    human_colors = [Color.RED, Color.ORANGE, Color.WHITE][: num_players - 1]

    console.print(Panel.fit(
        "[bold green]Catan RL — Real Life Mode[/bold green]\n"
        f"AI controls [bold blue]Blue[/bold blue]. "
        f"{num_players - 1} human player(s): "
        + ", ".join(c.value.capitalize() for c in human_colors),
        title="Welcome",
    ))

    model = MaskablePPO.load(model_path)
    console.print(f"[green]Model loaded from {model_path}[/green]")

    enemies = [HumanRelayPlayer(c) for c in human_colors]
    env = gymnasium.make("catanatron-v1", config={
        "map_type": "BASE",
        "vps_to_win": 10,
        "representation": "vector",
        "enemies": enemies,
    })

    obs, info = env.reset()
    u = env.unwrapped
    catan_map = u.game.state.board.map
    node_to_tiles = _build_node_to_tiles(catan_map)
    all_colors = [p.color for p in u.game.state.players]

    console.print("\n[bold]Game started![/bold]\n")

    while True:
        state = u.game.state
        _print_state_table(state, all_colors)

        winner = u.game.winning_color()
        if winner is not None:
            if winner == Color.BLUE:
                console.print(Panel("[bold green]AI WINS![/bold green]", title="Game Over"))
            else:
                console.print(Panel(
                    f"[bold red]{winner.value.capitalize()} wins![/bold red]",
                    title="Game Over",
                ))
            break

        # It's always the AI's turn here — the env auto-advanced opponents
        # by calling HumanRelayPlayer.decide() during the previous step().
        console.rule("[bold blue]AI's Turn[/bold blue]")

        valid = u.get_valid_actions()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[valid] = True
        action_int, _ = model.predict(obs, action_masks=mask, deterministic=True)
        action_int = int(action_int)
        decoded = from_action_space(action_int, state.playable_actions)

        description = action_to_english(decoded, catan_map, node_to_tiles)
        console.print(Panel(
            f"[bold yellow]>>> AI ACTION:[/bold yellow] {description}",
            title=f"Turn {state.num_turns}",
            border_style="yellow",
        ))

        if decoded.action_type == ActionType.ROLL:
            # Inject the real dice so resource production matches the physical board.
            console.print("[bold]Enter the dice the AI rolled on the table:[/bold]")
            d1, d2 = _prompt_dice()
            real_action = Action(decoded.color, ActionType.ROLL, (d1, d2))
            u.game.execute(real_action)
            u._advance_until_p0_decision()
            obs = u._get_observation()
            terminated = u.game.winning_color() is not None
            truncated = False
        else:
            console.input("[cyan]Press ENTER after executing this on the physical board...[/cyan] ")
            obs, _, terminated, truncated, _ = env.step(action_int)

        if terminated or truncated:
            continue

    env.close()
    console.print("\n[dim]Session ended.[/dim]")
