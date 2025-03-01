INSERT INTO algorithm_application_link
WITH classifications AS (
    SELECT
        UNNEST(algorithms) AS algorithm,
        application
    FROM classification
),

grouped_classifications AS (
    SELECT
        sim.group_id AS algorithm_id,
        sim.group_name AS algorithm_name,
        app.group_id AS application_id,
        app.group_name AS application_name,
    FROM classifications AS c
    LEFT JOIN algorithm_similarity AS sim ON c.algorithm = sim.value
    LEFT JOIN application_similarity AS app ON c.application = app.value
)

SELECT
    algorithm_id,
    application_id,
    algorithm_name,
    application_name,
    COUNT(*) AS score
FROM grouped_classifications
GROUP BY algorithm_id, application_id, algorithm_name, application_name
ORDER BY score DESC
