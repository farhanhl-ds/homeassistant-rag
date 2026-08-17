"""
Evaluates final LLM output quality by generating answers with prompt v1 and v2
for each golden question, then using an LLM judge to score each answer
1-5 on relevance/groundedness. Prints averages and writes eval/llm_results.json.

Run after eval/generate_golden_set.py.
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.rag import build_prompt, call_llm, retrieve  # noqa: E402

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.jsonl")
PROMPT_VERSIONS = ["v1", "v2"]
SAMPLE_SIZE = 25  # subset of golden set, keep eval fast/cheap

JUDGE_PROMPT = """You are evaluating a RAG system's answer for relevance and groundedness.

QUESTION: {question}

ANSWER: {answer}

Rate the ANSWER from 1 (irrelevant/hallucinated) to 5 (fully relevant and grounded).
Respond with ONLY a single integer 1-5."""


def load_golden_set():
    with open(GOLDEN_SET_PATH) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:SAMPLE_SIZE]


def judge(question: str, answer: str) -> int:
    raw = call_llm(JUDGE_PROMPT.format(question=question, answer=answer)).strip()
    digits = "".join(c for c in raw if c.isdigit())
    return int(digits[0]) if digits else 0


def main():
    golden = load_golden_set()
    if not golden:
        print("Golden set is empty. Run eval/generate_golden_set.py first.")
        return

    results = {v: [] for v in PROMPT_VERSIONS}

    for row in golden:
        chunks = retrieve(row["question"], mode="hybrid", top_k=5)
        for version in PROMPT_VERSIONS:
            prompt = build_prompt(row["question"], chunks, prompt_version=version)
            try:
                answer = call_llm(prompt)
                score = judge(row["question"], answer)
            except Exception as e:
                print(f"[warn] failed on {row['question'][:50]}: {e}")
                score = None
            results[version].append(score)

    print("\n=== LLM evaluation (prompt version comparison) ===")
    summary = {}
    for version, scores in results.items():
        valid = [s for s in scores if s is not None]
        avg = round(sum(valid) / len(valid), 3) if valid else 0
        summary[version] = {"avg_score": avg, "n": len(valid)}
        print(f"{version}: avg_score={avg} (n={len(valid)})")

    best = max(summary, key=lambda v: summary[v]["avg_score"])
    print(f"\nBest prompt version: {best} -> use this as the default in app/rag.py")

    out_path = os.path.join(os.path.dirname(__file__), "llm_results.json")
    with open(out_path, "w") as f:
        json.dump({"per_question": results, "summary": summary}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
