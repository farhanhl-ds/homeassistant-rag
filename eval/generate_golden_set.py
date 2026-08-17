"""
Generates a golden Q&A set for retrieval + LLM evaluation by asking the LLM
to invent a plausible user question for a sample of indexed chunks.

Output: eval/golden_set.jsonl with rows {doc_id, source, question}
"""
import json
import os
import random
import sys

import psycopg

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.rag import call_llm, get_pg_dsn  # noqa: E402

N_QUESTIONS_PER_DOC = 2
SAMPLE_SIZE = 60  # keep small so this finishes fast; raise once the pipeline works

QUESTION_GEN_PROMPT = """You are helping build a test set for a smart-home Q&A RAG system.
Given the DOCUMENT below, write {n} distinct, realistic questions that a user could ask
whose answer is fully contained in this document. Return ONLY the questions, one per line,
no numbering.

DOCUMENT:
{content}
"""


def main():
    with psycopg.connect(get_pg_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT doc_id, source, content FROM documents")
            rows = cur.fetchall()

    if not rows:
        print("No documents found. Run ingestion/ingest.py first.")
        return

    sample = random.sample(rows, min(SAMPLE_SIZE, len(rows)))

    out_path = os.path.join(os.path.dirname(__file__), "golden_set.jsonl")
    with open(out_path, "w") as f:
        for doc_id, source, content in sample:
            prompt = QUESTION_GEN_PROMPT.format(n=N_QUESTIONS_PER_DOC, content=content[:1500])
            try:
                raw = call_llm(prompt)
            except Exception as e:
                print(f"[warn] generation failed for {doc_id}: {e}")
                continue
            for line in raw.strip().split("\n"):
                line = line.strip("- ").strip()
                if not line:
                    continue
                f.write(json.dumps({"doc_id": doc_id, "source": source, "question": line}) + "\n")

    print(f"Wrote golden set to {out_path}")


if __name__ == "__main__":
    main()
