CREATE TABLE IF NOT EXISTS classification (
    doi VARCHAR PRIMARY KEY,
    datasets VARCHAR[],
    algorithms VARCHAR[],
    application VARCHAR,
);
