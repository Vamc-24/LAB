const express = require('express');
const app = express();
app.get('/', (req, res) => {
    res.send(`
        <script>
            console.log("Hello World");
        </script>
    `);
});
app.listen(3000, () => {
    console.log('Server running on http://localhost:3000');
});
