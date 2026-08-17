"""
Core retrieval + generation logic.

Previously (see git history) this loaded data/corpus.json + embeddings.npy into
an in-memory minsearch.VectorSearch index which is semantic search only. Swapped 
to Postgres (pgvector + tsvector) so we can do lexical search too and combine both
into hybrid search via Reciprocal Rank Fusion (RRF).
"""
import os

import psycopg
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq
from pgvector.psycopg import register_vector

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# LLM_MODEL = "llama-3.1-8b-instant"
LLM_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about Home Assistant, \
Zigbee2MQTT, and ESPHome using the CONTEXT below. Only use the context; if the answer \
isn't there, say you don't know.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedder


def get_pg_dsn() -> str:
    return (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', 5432)} "
        f"dbname={os.getenv('POSTGRES_DB', 'homeassistant_rag')} "
        f"user={os.getenv('POSTGRES_USER', 'raguser')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'ragpass')}"
    )


def embed_query(query: str) -> list[float]:
    return list(get_embedder().embed([query]))[0].tolist()


def retrieve(query: str, mode: str = "hybrid", top_k: int = 5) -> list[dict]:
    """mode: 'vector' (semantic only) | 'text' (lexical only) | 'hybrid' (both, via RRF)"""
    q_emb = embed_query(query)

    with psycopg.connect(get_pg_dsn()) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            if mode == "vector":
                cur.execute(
                    """
                    SELECT doc_id, source, source_url, title, content
                    FROM documents
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q_emb, top_k),
                )
            elif mode == "text":
                cur.execute(
                    """
                    SELECT doc_id, source, source_url, title, content
                    FROM documents
                    WHERE tsv @@ plainto_tsquery('english', %s)
                    ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %s)) DESC
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
                    SELECT d.doc_id, d.source, d.source_url, d.title, d.content
                    FROM fused f JOIN documents d ON d.doc_id = f.doc_id
                    ORDER BY f.score DESC
                    LIMIT %s
                    """,
                    (q_emb, q_emb, query, query, query, top_k),
                )
            rows = cur.fetchall()

    cols = ["doc_id", "source", "source_url", "title", "content"]
    return [dict(zip(cols, r)) for r in rows]


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


def answer_question(question: str, mode: str = "hybrid", top_k: int = 5) -> dict:
    chunks = retrieve(question, mode=mode, top_k=top_k)
    prompt = build_prompt(question, chunks)
    answer = call_llm(prompt)

    seen_urls = set()
    sources = []
    for c in chunks:
        if c["source_url"] not in seen_urls:
            seen_urls.add(c["source_url"])
            sources.append({"url": c["source_url"], "title": c["title"]})

    return {
        "question": question,
        "answer": answer,
        "retrieval_mode": mode,
        "sources": sources,
    }
