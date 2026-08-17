"""
Evaluates retrieval quality of vector-only, text-only, and hybrid search against
the golden Q&A set, using Hit Rate@5 and MRR@5. Prints a comparison table and
writes results to eval/retrieval_results.json.

Run after eval/generate_golden_set.py.
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.rag import retrieve  # noqa: E402

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.jsonl")
MODES = ["vector", "text", "hybrid"]
TOP_K = 5


def load_golden_set():
    with open(GOLDEN_SET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def hit_rate_and_mrr(golden, mode):
    hits = 0
    reciprocal_ranks = []
    for row in golden:
        results = retrieve(row["question"], mode=mode, top_k=TOP_K)
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

    results = {}
    for mode in MODES:
        print(f"Evaluating mode={mode} ...")
        results[mode] = hit_rate_and_mrr(golden, mode)

    print("\n=== Retrieval evaluation (Top-{}) ===".format(TOP_K))
    print(f"{'mode':<10} {'hit_rate':<10} {'mrr':<10} n")
    for mode, m in results.items():
        print(f"{mode:<10} {m['hit_rate']:<10} {m['mrr']:<10} {m['n']}")

    best = max(results, key=lambda m: results[m]["mrr"])
    print(f"\nBest mode by MRR: {best} -> use this as the default in app/rag.py / app/main.py")

    out_path = os.path.join(os.path.dirname(__file__), "retrieval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
