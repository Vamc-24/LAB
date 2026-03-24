const express = require('express');
const app = express();
app.get('/', (req, res) => {
    res.send('Welcome to Home Page');
});
app.get('/about', (req, res) => {
    res.send('This is About Page');
});
app.get('/contact', (req, res) => {
    res.send('Contact Us at contact@example.com');
});
app.listen(3000, () => {
    console.log('Website running on http://localhost:3000');
});
