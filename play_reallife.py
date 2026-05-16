#!/usr/bin/env python3
"""Real-life Catan mode: AI announces moves, human executes them on a physical board.

Usage:
    python play_reallife.py --model runs/.../best_model.zip
    python play_reallife.py --model runs/.../best_model.zip --players 3
"""

import argparse
from reallife.cli import run_reallife


def main():
    parser = argparse.ArgumentParser(
        description="Play Catan with an AI agent on a real physical board"
    )
    parser.add_argument("--model", required=True, help="Path to trained model (.zip)")
    parser.add_argument("--players", type=int, default=4,
                        help="Total number of players (1 AI + rest human)")
    args = parser.parse_args()

    if args.players < 2 or args.players > 4:
        print("Error: --players must be between 2 and 4")
        raise SystemExit(1)

    run_reallife(model_path=args.model, num_players=args.players)


if __name__ == "__main__":
    main()
