import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.py"

BOT_FILES = [
    "./ql_bot.py",
    "./meta_ql_bot.py",
    "./ultra_bot.py",
    "./apex_bot.py",
    "./icm_bot.py",
    "./nested_bot.py",
    "./example_bot.py",
]

ENGINE_CMD = [".venv/bin/python", "engine.py", "--small_log"]


def read_config_text() -> str:
    return CONFIG_PATH.read_text(encoding="utf-8")


def write_config_text(text: str) -> None:
    CONFIG_PATH.write_text(text, encoding="utf-8")


def update_config(bot_a: str, bot_b: str) -> None:
    text = read_config_text()
    text = re.sub(r"^BOT_1_NAME\s*=.*$", "BOT_1_NAME = \"BotA\"", text, flags=re.M)
    text = re.sub(r"^BOT_2_NAME\s*=.*$", "BOT_2_NAME = \"BotB\"", text, flags=re.M)
    text = re.sub(r"^BOT_1_FILE\s*=.*$", f"BOT_1_FILE = \"{bot_a}\"", text, flags=re.M)
    text = re.sub(r"^BOT_2_FILE\s*=.*$", f"BOT_2_FILE = \"{bot_b}\"", text, flags=re.M)
    write_config_text(text)


def run_match() -> tuple[int, int]:
    proc = subprocess.run(ENGINE_CMD, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    bot_a_bankroll = None
    bot_b_bankroll = None
    for line in out.splitlines():
        if line.strip().startswith("Total Bankroll:"):
            if bot_a_bankroll is None:
                bot_a_bankroll = int(line.split(":", 1)[1].strip())
            else:
                bot_b_bankroll = int(line.split(":", 1)[1].strip())
                break
    if bot_a_bankroll is None or bot_b_bankroll is None:
        raise RuntimeError("Failed to parse bankrolls from engine output")
    return bot_a_bankroll, bot_b_bankroll


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <candidate_bot>")
        return 2

    candidate = sys.argv[1]
    opponents = [b for b in BOT_FILES if b != candidate]

    original = read_config_text()
    results = {}
    try:
        for opp in opponents:
            wins = 0
            losses = 0
            matches = 5
            for _ in range(matches):
                update_config(candidate, opp)
                a_bankroll, b_bankroll = run_match()
                if a_bankroll > b_bankroll:
                    wins += 1
                else:
                    losses += 1
            results[opp] = (wins, losses)
    finally:
        write_config_text(original)

    print("\n=== Benchmark Results ===")
    for opp, (wins, losses) in results.items():
        print(f"{candidate} vs {opp}: {wins}-{losses}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
