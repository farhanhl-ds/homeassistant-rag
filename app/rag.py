"""
Core retrieval + generation logic. Kept separate from the FastAPI/Streamlit
layer so eval/ scripts can import and reuse it directly.
"""
import os
import time

import psycopg
from dotenv import load_dotenv
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

_embedder = None
_reranker = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


def get_pg_dsn() -> str:
    return (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', 5432)} "
        f"dbname={os.getenv('POSTGRES_DB', 'homeassistant_rag')} "
        f"user={os.getenv('POSTGRES_USER', 'raguser')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'ragpass')}"
    )


# ---------- Retrieval ----------

def embed_query(query: str) -> list[float]:
    return list(get_embedder().embed([query]))[0].tolist()


def retrieve(query: str, mode: str = "hybrid", top_k: int = 5) -> list[dict]:
    """mode: 'vector' | 'text' | 'hybrid'"""
    q_emb = embed_query(query)

    with psycopg.connect(get_pg_dsn()) as conn:
        with conn.cursor() as cur:
            if mode == "vector":
                cur.execute(
                    """
                    SELECT doc_id, source, source_url, title, content,
                           1 - (embedding <=> %s::vector) AS score
                    FROM documents
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q_emb, q_emb, top_k),
                )
            elif mode == "text":
                cur.execute(
                    """
                    SELECT doc_id, source, source_url, title, content,
                           ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score
                    FROM documents
                    WHERE tsv @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, query, top_k),
                )
            else:  # hybrid: reciprocal rank fusion over vector + text results
                cur.execute(
                    """
                    WITH vec AS (
                        SELECT doc_id, row_number() OVER (ORDER BY embedding <=> %s::vector) AS rnk
                        FROM documents ORDER BY embedding <=> %s::vector LIMIT 20
                    ),
                    txt AS (
                        SELECT doc_id, row_number() OVER (ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %s)) DESC) AS rnk
                        FROM documents WHERE tsv @@ plainto_tsquery('english', %s)
                        ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %s)) DESC LIMIT 20
                    ),
                    fused AS (
                        SELECT doc_id, SUM(1.0 / (60 + rnk)) AS score
                        FROM (SELECT * FROM vec UNION ALL SELECT * FROM txt) u
                        GROUP BY doc_id
                    )
                    SELECT d.doc_id, d.source, d.source_url, d.title, d.content, f.score
                    FROM fused f JOIN documents d ON d.doc_id = f.doc_id
                    ORDER BY f.score DESC
                    LIMIT %s
                    """,
                    (q_emb, q_emb, query, query, query, top_k),
                )
            rows = cur.fetchall()

    cols = ["doc_id", "source", "source_url", "title", "content", "score"]
    return [dict(zip(cols, r)) for r in rows]


# ---------- Reranking (best-practice rubric item) ----------

def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Re-scores candidate chunks with a cross-encoder (more accurate, slower
    than the bi-encoder/text search used for initial retrieval) and returns
    the top_k best matches."""
    if not chunks:
        return chunks
    documents = [c["content"] for c in chunks]
    scores = list(get_reranker().rerank(query, documents))
    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


def retrieve_reranked(query: str, mode: str = "hybrid", candidate_k: int = 20,
                       top_k: int = 5) -> list[dict]:
    """Retrieve a larger candidate pool, then rerank down to top_k."""
    candidates = retrieve(query, mode=mode, top_k=candidate_k)
    return rerank(query, candidates, top_k=top_k)


# ---------- Query rewriting (best-practice rubric item) ----------

def rewrite_query(raw_query: str, history: list[str] | None = None) -> str:
    """Cheap query rewrite: expand abbreviations / make standalone. Falls back
    to the raw query if the LLM call fails, so retrieval never breaks."""
    try:
        prompt = (
            "Rewrite the user question into a clear, standalone search query "
            "for a smart-home documentation search engine. Keep it short, no preamble.\n\n"
            f"Question: {raw_query}\nRewritten query:"
        )
        return call_llm(prompt, model=LLM_MODEL).strip()
    except Exception:
        return raw_query


# ---------- Prompt building ----------

PROMPT_TEMPLATE_V1 = """You are a helpful assistant answering questions about Home Assistant, \
Zigbee2MQTT, and ESPHome using the CONTEXT below. Only use the context; if the answer \
isn't there, say you don't know.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

PROMPT_TEMPLATE_V2 = """You are a senior smart-home support engineer. Answer the QUESTION \
strictly using the CONTEXT. Be concise, use bullet points for steps, and cite which \
source (home_assistant / zigbee2mqtt / esphome) each fact comes from. If the context is \
insufficient, say so explicitly instead of guessing.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

PROMPTS = {"v1": PROMPT_TEMPLATE_V1, "v2": PROMPT_TEMPLATE_V2}


def build_prompt(question: str, chunks: list[dict], prompt_version: str = "v2") -> str:
    context = "\n\n".join(f"[{c['source']}] {c['content']}" for c in chunks)
    return PROMPTS[prompt_version].format(context=context, question=question)


# ---------- LLM call ----------

def call_llm(prompt: str, model: str = LLM_MODEL) -> str:
    if LLM_PROVIDER == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    if LLM_PROVIDER == "ollama":
        import requests
        resp = requests.post(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


# ---------- End-to-end ----------

def answer_question(question: str, mode: str = "hybrid", prompt_version: str = "v2",
                     use_rerank: bool = False, rewrite: bool = True) -> dict:
    t0 = time.time()
    search_query = rewrite_query(question) if rewrite else question
    if use_rerank:
        chunks = retrieve_reranked(search_query, mode=mode)
    else:
        chunks = retrieve(search_query, mode=mode)
    prompt = build_prompt(question, chunks, prompt_version=prompt_version)
    answer = call_llm(prompt)
    return {
        "question": question,
        "search_query": search_query,
        "answer": answer,
        "retrieval_mode": mode,
        "prompt_version": prompt_version,
        "rerank": use_rerank,
        "model": LLM_MODEL,
        "retrieved_ids": [c["doc_id"] for c in chunks],
        "sources": [{"url": c["source_url"], "title": c["title"]} for c in chunks],
        "response_time_s": round(time.time() - t0, 3),
    }
