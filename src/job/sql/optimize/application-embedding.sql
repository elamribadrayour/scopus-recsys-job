DROP INDEX IF EXISTS application_embedding_embedding_index;
CREATE INDEX application_embedding_embedding_index
ON application_embedding
USING HNSW (embedding)
WITH (metric = 'cosine');
