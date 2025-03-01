-- Create an optimized version of the algorithms similarity table by ordering by group id
CREATE OR REPLACE TABLE algorithm_similarity_clustered AS
SELECT *
FROM algorithm_similarity
ORDER BY group_id;

-- Swap the tables so that the new optimized table becomes the public 'leads' table.
ALTER TABLE algorithm_similarity RENAME TO algorithm_similarity_backup;
ALTER TABLE algorithm_similarity_clustered RENAME TO algorithm_similarity;
DROP TABLE algorithm_similarity_backup;

-- Create an index on the group id column.
CREATE INDEX algorithm_group_id_idx ON algorithm_similarity(group_id);
