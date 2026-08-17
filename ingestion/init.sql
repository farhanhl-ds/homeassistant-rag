CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
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
