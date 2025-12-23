
"""Batch helper to run media + validation in one go.

Usage:
  python -m agents.batch_agent --limit 50
"""
import argparse
from . import media_agent, validate_agent


def main(limit: int = 50):
    print("[BATCH] Running media_agent...")
    media_agent.main(limit=limit)
    print("[BATCH] Running validate_agent...")
    validate_agent.main(limit=limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    main(limit=args.limit)
