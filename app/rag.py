"""
Core retrieval + generation logic
Refactored from notebooks/01_mvp_prototype.ipynb.

Loads the corpus + embeddings saved by ingestion/ingest.py, builds an in-memory
minsearch.VectorSearch index, and exposes answer_question() for the app/eval
layers to reuse.
"""
import json
import os

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq
from minsearch import VectorSearch

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about Home Assistant, \
Zigbee2MQTT, and ESPHome using the CONTEXT below. Only use the context; if the answer \
isn't there, say you don't know.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

_embedder = None
_vindex = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedder


def get_index():
    """Lazily loads corpus.json + embeddings.npy (produced by ingestion/ingest.py)
    and builds the in-memory vector index once per process."""
    global _vindex
    if _vindex is None:
        corpus_path = os.path.join(DATA_DIR, "corpus.json")
        embeddings_path = os.path.join(DATA_DIR, "embeddings.npy")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"{corpus_path} not found - run `uv run python ingestion/ingest.py` first"
            )
        with open(corpus_path) as f:
            corpus = json.load(f)
        embeddings = np.load(embeddings_path)

        _vindex = VectorSearch(keyword_fields=[])
        _vindex.fit(embeddings, corpus)
    return _vindex


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    q_emb = list(get_embedder().embed([query]))[0]
    return get_index().search(q_emb, num_results=top_k)


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{c['source']}] {c['content']}" for c in chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def call_llm(prompt: str) -> str:
    client = groq_client
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def answer_question(question: str, top_k: int = 5) -> dict:
    chunks = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, chunks)
    answer = call_llm(prompt)
    return {
        "question": question,
        "answer": answer,
        "sources": [{"url": c["url"], "title": c["title"]} for c in chunks],
    }
