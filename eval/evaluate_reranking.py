"""
Evaluates whether adding a cross-encoder reranker on top of hybrid retrieval
improves Hit Rate@5 / MRR@5 vs hybrid retrieval alone. Same golden set and
metric as eval/evaluate_retrieval.py, so results are directly comparable.

Run after eval/generate_golden_set.py.
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.rag import retrieve, retrieve_reranked  # noqa: E402

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.jsonl")
TOP_K = 5


def load_golden_set():
    with open(GOLDEN_SET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def hit_rate_and_mrr(golden, use_rerank: bool):
    hits = 0
    reciprocal_ranks = []
    for row in golden:
        if use_rerank:
            results = retrieve_reranked(row["question"], mode="hybrid", top_k=TOP_K)
        else:
            results = retrieve(row["question"], mode="hybrid", top_k=TOP_K)
        ids = [r["doc_id"] for r in results]
        if row["doc_id"] in ids:
            hits += 1
            rank = ids.index(row["doc_id"]) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
    n = len(golden)
    return {
        "hit_rate": round(hits / n, 4) if n else 0,
        "mrr": round(sum(reciprocal_ranks) / n, 4) if n else 0,
        "n": n,
    }


def main():
    golden = load_golden_set()
    if not golden:
        print("Golden set is empty. Run eval/generate_golden_set.py first.")
        return

    print("Evaluating hybrid (no rerank) ...")
    plain = hit_rate_and_mrr(golden, use_rerank=False)
    print("Evaluating hybrid + rerank ...")
    reranked = hit_rate_and_mrr(golden, use_rerank=True)

    print(f"\n=== Reranking evaluation (Top-{TOP_K}) ===")
    print(f"{'variant':<20} {'hit_rate':<10} {'mrr':<10} n")
    print(f"{'hybrid':<20} {plain['hit_rate']:<10} {plain['mrr']:<10} {plain['n']}")
    print(f"{'hybrid + rerank':<20} {reranked['hit_rate']:<10} {reranked['mrr']:<10} {reranked['n']}")

    winner = "hybrid + rerank" if reranked["mrr"] > plain["mrr"] else "hybrid (no rerank)"
    print(f"\nBest variant by MRR: {winner}")

    out_path = os.path.join(os.path.dirname(__file__), "rerank_results.json")
    with open(out_path, "w") as f:
        json.dump({"hybrid": plain, "hybrid_rerank": reranked}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
