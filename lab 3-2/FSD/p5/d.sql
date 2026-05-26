-- Create Database
CREATE DATABASE project_db;
USE project_db;
-- Create Table
CREATE TABLE project (
    project_id INT PRIMARY KEY,
    project_name VARCHAR(50),
    duration INT
);
-- Insert Data
INSERT INTO project
VALUES (101, 'AI System', 6);
INSERT INTO project
VALUES (102, 'Web Portal', 4);
-- Display Data
SELECT *
FROM project;