CREATE TABLE IF NOT EXISTS algorithm_embedding (
    algorithm VARCHAR PRIMARY KEY,
    embedding FLOAT[{model_dimension}]
);
