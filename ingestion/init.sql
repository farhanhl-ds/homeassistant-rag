CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,          -- stable id, e.g. hash of source_url+chunk_idx
    source      TEXT NOT NULL,          -- "home_assistant" | "zigbee2mqtt" | "esphome"
    source_url  TEXT NOT NULL,
    title       TEXT,
    chunk_idx   INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(384),
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS documents_doc_id_idx ON documents (doc_id);
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents
    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS documents_tsv_idx ON documents USING gin (tsv);

-- conversation log used for LLM eval + monitoring
CREATE TABLE IF NOT EXISTS conversations (
    id              BIGSERIAL PRIMARY KEY,
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    retrieval_mode  TEXT NOT NULL,      -- "vector" | "text" | "hybrid"
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    retrieved_ids   TEXT[],
    response_time_s DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT REFERENCES conversations(id),
    rating          SMALLINT NOT NULL,  -- 1 = thumbs up, -1 = thumbs down
    created_at      TIMESTAMPTZ DEFAULT now()
);
