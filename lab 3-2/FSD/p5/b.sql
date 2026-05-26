-- Create Table
CREATE TABLE employees ( 
emp_id INT PRIMARY KEY, 
emp_name VARCHAR(50), 
salary INT
);
-- Insert Data
INSERT INTO employees VALUES (1, 'Ravi', 25000);
INSERT INTO employees VALUES (2, 'Anjali', 30000);
-- Update Data 
UPDATE employees 
SET salary = 35000 
WHERE emp_id = 2;
-- Display Data
SELECT * FROM employees;
