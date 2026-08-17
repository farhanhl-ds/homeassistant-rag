import os

import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag import answer_question, get_pg_dsn

app = FastAPI(title="Home Assistant RAG API")


class AskRequest(BaseModel):
    question: str
    mode: str = "hybrid"          # vector | text | hybrid
    prompt_version: str = "v2"    # v1 | v2
    rerank: bool = True


class FeedbackRequest(BaseModel):
    conversation_id: int
    rating: int  # 1 or -1


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    if req.mode not in ("vector", "text", "hybrid"):
        raise HTTPException(400, "mode must be vector, text, or hybrid")

    result = answer_question(req.question, mode=req.mode, prompt_version=req.prompt_version,
                              use_rerank=req.rerank)

    with psycopg.connect(get_pg_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations
                    (question, answer, retrieval_mode, model, prompt_version, retrieved_ids, response_time_s)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (result["question"], result["answer"], result["retrieval_mode"],
                 result["model"], result["prompt_version"], result["retrieved_ids"],
                 result["response_time_s"]),
            )
            conversation_id = cur.fetchone()[0]
        conn.commit()

    result["conversation_id"] = conversation_id
    return result


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.rating not in (1, -1):
        raise HTTPException(400, "rating must be 1 or -1")
    with psycopg.connect(get_pg_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (conversation_id, rating) VALUES (%s, %s)",
                (req.conversation_id, req.rating),
            )
        conn.commit()
    return {"status": "ok"}
