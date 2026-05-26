-- Create Database
CREATE DATABASE api_db;
USE api_db;
-- Create Table for API Integration 
CREATE TABLE users (
user_id INT AUTO_INCREMENT PRIMARY KEY, 
username VARCHAR(50) NOT NULL,
password VARCHAR(100) NOT NULL, 
email VARCHAR(100)
);
