const express = require('express');
const mysql = require('mysql2');
const app = express();

app.use(express.json());

// 🔹 MySQL Connection
const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: '9922', // ⚠ Put your MySQL password
    database: 'testdb'
});

// 🔹 Connect to MySQL (With Proper Error Debugging)
db.connect((err) => {
    if (err) {
        console.log('Database connection failed');
        console.log('Error Details:', err.message);
    } else {
        console.log('MySQL Connected Successfully');
    }
});

// 🔹 Home Route
app.get('/', (req, res) => {
    res.send('Server + MySQL is running');
});

// 🔹 Get All Students
app.get('/students', (req, res) => {
    db.query('SELECT * FROM students', (err, result) => {
        if (err) {
            console.log('Query Error:', err.message);
            res.status(500).send('Database Query Failed');
        } else {
            res.json(result);
        }
    });
});

// 🔹 Start Server
app.listen(3000, () => {
    console.log('Server running on http://localhost:3000');
});