"""
Ingestion script.

Scrapes the doc pages listed in sources.yaml, chunks them, embeds each chunk
locally with fastembed, and upserts into Postgres (pgvector + tsvector, so we
can do both semantic and lexical/hybrid search - see app/rag.py).

Previously (see git history) this saved to data/corpus.json + embeddings.npy
for an in-memory minsearch index. Swapped to Postgres so we can add lexical
search and evaluate vector vs. text vs. hybrid retrieval properly.

Run:
    uv run python ingestion/ingest.py
"""
import hashlib
import os
import time

import psycopg
import requests
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector
from tqdm import tqdm

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

HERE = os.path.dirname(__file__)


def get_pg_dsn() -> str:
    return (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', 5432)} "
        f"dbname={os.getenv('POSTGRES_DB', 'homeassistant_rag')} "
        f"user={os.getenv('POSTGRES_USER', 'raguser')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'ragpass')}"
    )


def fetch_page(url: str) -> str:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "homeassistant-rag-bot/0.1"})
    resp.raise_for_status()
    return resp.text


def html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    main = soup.find("main") or soup.find("article") or soup.body or soup
    for tag in main.find_all(["nav", "script", "style", "footer", "header"]):
        tag.decompose()
    return title, main.get_text(separator="\n", strip=True)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def load_sources() -> list[dict]:
    with open(os.path.join(HERE, "sources.yaml")) as f:
        raw = yaml.safe_load(f)
    items = []
    for source, cfg in raw.items():
        for page in cfg["pages"]:
            items.append({"source": source, "url": cfg["base_url"].rstrip("/") + page})
    return items


def run():
    sources = load_sources()
    corpus = []

    for item in tqdm(sources, desc="scraping"):
        try:
            html = fetch_page(item["url"])
        except Exception as e:
            print(f"[warn] failed to fetch {item['url']}: {e}")
            continue
        title, text = html_to_text(html)
        for idx, chunk in enumerate(chunk_text(text)):
            doc_id = hashlib.sha256(f"{item['url']}::{idx}".encode()).hexdigest()
            corpus.append({
                "doc_id": doc_id,
                "source": item["source"],
                "source_url": item["url"],
                "title": title,
                "chunk_idx": idx,
                "content": chunk,
            })
        time.sleep(0.3)

    print(f"Total chunks: {len(corpus)}")

    print("Embedding chunks...")
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    texts = [c["content"] for c in corpus]
    embeddings = list(embedder.embed(texts))

    print("Writing to Postgres...")
    with psycopg.connect(get_pg_dsn()) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for c, emb in tqdm(list(zip(corpus, embeddings)), desc="upserting"):
                cur.execute(
                    """
                    INSERT INTO documents (doc_id, source, source_url, title, chunk_idx, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding
                    """,
                    (c["doc_id"], c["source"], c["source_url"], c["title"],
                     c["chunk_idx"], c["content"], list(emb)),
                )
        conn.commit()

    print(f"Done. Upserted {len(corpus)} chunks into Postgres.")


if __name__ == "__main__":
    run()
