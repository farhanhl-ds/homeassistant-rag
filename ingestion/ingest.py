"""
Refactored ingestion script from notebooks/01_mvp_prototype.ipynb.

Scrapes the doc pages listed in sources.yaml, chunks them, embeds each chunk
locally with fastembed, and saves the result to data/ so app/rag.py can load
it without re-scraping every time.

Still in-memory/file-based at this stage (no database yet which will later be
committed once we move to Postgres+pgvector).

Run:
    uv run python ingestion/ingest.py
"""
import json
import os
import time

import numpy as np
import requests
import yaml
from bs4 import BeautifulSoup
from fastembed import TextEmbedding
from tqdm import tqdm

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")


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
            corpus.append({
                "source": item["source"],
                "url": item["url"],
                "title": title,
                "chunk_idx": idx,
                "content": chunk,
            })
        time.sleep(0.3)

    print(f"Total chunks: {len(corpus)}")

    print("Embedding chunks...")
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    texts = [c["content"] for c in corpus]
    embeddings = np.array(list(embedder.embed(texts)))

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "corpus.json"), "w") as f:
        json.dump(corpus, f)
    np.save(os.path.join(DATA_DIR, "embeddings.npy"), embeddings)

    print(f"Saved {len(corpus)} chunks + embeddings {embeddings.shape} to {DATA_DIR}")


if __name__ == "__main__":
    run()
