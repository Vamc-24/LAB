const express = require('express');
const app = express();

app.use(express.json());

let users = [];

app.get('/', (req, res) => {
    res.send('CRUD API is running');
});

// CREATE
app.post('/users', (req, res) => {
    users.push(req.body);
    res.send('User added');
});

// READ
app.get('/users', (req, res) => {
    res.json(users);
});

// UPDATE
app.put('/users/:id', (req, res) => {
    users[req.params.id] = req.body;
    res.send('User updated');
});

// DELETE
app.delete('/users/:id', (req, res) => {
    users.splice(req.params.id, 1);
    res.send('User deleted');
});

app.listen(3000, () => {
    console.log('CRUD server running on http://localhost:3000');
});