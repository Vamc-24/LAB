-- Create Table
CREATE TABLE marks(
    student_id INT,
    student_name VARCHAR(50),
    score INT
);
-- Insert Data
INSERT INTO marks
VALUES (1, 'Sai', 80);
INSERT INTO marks
VALUES (2, 'Priya', 90);
INSERT INTO marks
VALUES (3, 'Rahul', 70);
-- Subquery Example 
SELECT student_name,
    score
FROM marks
WHERE score > (
        SELECT AVG(score)
        FROM marks
    );