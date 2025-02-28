DROP INDEX IF EXISTS algorithm_embedding_embedding_index;
CREATE INDEX algorithm_embedding_embedding_index
ON algorithm_embedding
USING HNSW (embedding)
WITH (metric = 'cosine');
