"""
Quick demo of query rewriting: shows the raw user question next to what the
LLM rewrites it into before retrieval. Not a formal eval (see README for why),
just illustrative examples for documentation.

Run:
    docker compose exec app python eval/demo_query_rewrite.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.rag import rewrite_query  # noqa: E402

EXAMPLE_QUESTIONS = [
    "kok device gua unavailable terus ya",
    "gimana cara nge-pair-nya?",
    "why won't it stay connected",
    "how do i make it turn on automatically at sunset",
    "esphome ota not working help",
]


def main():
    print(f"{'RAW QUESTION':<55} | REWRITTEN QUERY")
    print("-" * 110)
    for q in EXAMPLE_QUESTIONS:
        rewritten = rewrite_query(q)
        print(f"{q:<55} | {rewritten}")


if __name__ == "__main__":
    main()
