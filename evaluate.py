#!/usr/bin/env python3
"""Evaluate a trained Catan RL model against various opponents.

Usage:
    python evaluate.py --model runs/.../best_model.zip
    python evaluate.py --model runs/.../best_model.zip --opponents random weighted --games 200
    python evaluate.py --model runs/.../best_model.zip --games 100 --num-players 4
"""

import argparse
import os
import sys
import numpy as np
import gymnasium
from rich.console import Console
from rich.table import Table
from rich import box

from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_gym.envs.catanatron_env import ACTION_SPACE_SIZE
from catanatron.state_functions import get_actual_victory_points

console = Console()

OPPONENT_COLORS = [Color.RED, Color.ORANGE, Color.WHITE]


def build_opponents(name: str, num_players: int):
    colors = OPPONENT_COLORS[: num_players - 1]
    if name == "random":
        return [RandomPlayer(c) for c in colors]
    if name == "weighted":
        return [WeightedRandomPlayer(c) for c in colors]
    raise ValueError(f"Unknown opponent: {name}")


def evaluate(model, opponent_name: str, num_players: int, n_games: int, verbose: bool = False):
    """Run n_games and return stats dict."""
    enemies = build_opponents(opponent_name, num_players)
    env = gymnasium.make("catanatron-v1", config={
        "map_type": "BASE",
        "vps_to_win": 10,
        "representation": "vector",
        "enemies": enemies,
    })

    wins = 0
    losses = 0
    draws = 0
    vps = []
    lengths = []

    for game_idx in range(n_games):
        obs, info = env.reset(seed=game_idx)
        u = env.unwrapped
        done = False
        steps = 0

        while not done:
            valid = u.get_valid_actions()
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
            mask[valid] = True
            action_int, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action_int))
            done = terminated or truncated
            steps += 1

        winner = u.game.winning_color()
        vp = get_actual_victory_points(u.game.state, Color.BLUE)
        vps.append(vp)
        lengths.append(steps)

        if winner == Color.BLUE:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1

        if verbose and (game_idx + 1) % 20 == 0:
            console.print(f"  Game {game_idx+1}/{n_games}: W={wins} L={losses} D={draws}")

    env.close()

    return {
        "opponent": opponent_name,
        "n_games": n_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_games,
        "loss_rate": losses / n_games,
        "draw_rate": draws / n_games,
        "mean_vp": float(np.mean(vps)),
        "std_vp": float(np.std(vps)),
        "mean_length": float(np.mean(lengths)),
    }


def print_results(all_stats: list):
    table = Table(title="Evaluation Results", box=box.ROUNDED)
    table.add_column("Opponent", style="bold")
    table.add_column("Games")
    table.add_column("Win %", style="green")
    table.add_column("Loss %", style="red")
    table.add_column("Draw %")
    table.add_column("Mean VP")
    table.add_column("Mean Steps")

    for s in all_stats:
        table.add_row(
            s["opponent"],
            str(s["n_games"]),
            f"{s['win_rate']:.1%}",
            f"{s['loss_rate']:.1%}",
            f"{s['draw_rate']:.1%}",
            f"{s['mean_vp']:.1f} ± {s['std_vp']:.1f}",
            f"{s['mean_length']:.0f}",
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Catan RL model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.zip)")
    parser.add_argument("--opponents", nargs="+", default=["random", "weighted"],
                        choices=["random", "weighted"], help="Opponent types to evaluate against")
    parser.add_argument("--games", type=int, default=100, help="Number of games per opponent")
    parser.add_argument("--num-players", type=int, default=4, help="Total players (1 AI + rest)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON results to stdout (for autoresearch)")
    args = parser.parse_args()

    from sb3_contrib.ppo_mask import MaskablePPO
    console.print(f"[bold]Loading model:[/bold] {args.model}")
    model = MaskablePPO.load(args.model)

    all_stats = []
    for opp in args.opponents:
        console.print(f"\n[bold]Evaluating vs {opp}...[/bold] ({args.games} games)")
        stats = evaluate(model, opp, args.num_players, args.games, verbose=args.verbose)
        all_stats.append(stats)
        console.print(f"  Win rate: [green]{stats['win_rate']:.1%}[/green] | Mean VP: {stats['mean_vp']:.1f}")

    console.print()
    print_results(all_stats)

    if args.json:
        import json as _json
        print(_json.dumps(all_stats))


if __name__ == "__main__":
    main()
