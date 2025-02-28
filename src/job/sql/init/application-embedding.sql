CREATE TABLE IF NOT EXISTS application_embedding (
    application VARCHAR PRIMARY KEY,
    embedding FLOAT[{model_dimension}]
);
