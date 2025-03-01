-- Create an optimized version of the application similarity table by ordering by group id
CREATE OR REPLACE TABLE application_similarity_clustered AS
SELECT *
FROM application_similarity
ORDER BY group_id;

-- Swap the tables so that the new optimized table becomes the public 'application_similarity' table.
ALTER TABLE application_similarity RENAME TO application_similarity_backup;
ALTER TABLE application_similarity_clustered RENAME TO application_similarity;
DROP TABLE application_similarity_backup;

-- Create an index on the group id column.
CREATE INDEX application_group_id_idx ON application_similarity(group_id);
