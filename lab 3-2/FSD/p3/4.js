import React, { useState } from "react";
function App() {
    const [name, setName] = useState("");
    const [email, setEmail] = useState("");
    const handleSubmit = (event) => {
        event.preventDefault();
        alert(`Submitted Details:\nName: ${name}\nEmail: ${email}`);
    };
    return (
        <div style={{ textAlign: "center", marginTop: "50px" }}>
            <h2>React Form Example</h2>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    placeholder="Enter Name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                />
                <br /><br />
                <input
                    type="email"
                    placeholder="Enter Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                />
                <br /><br />
                <button type="submit">Submit</button>
            </form>
        </div>
    );
}
export default App;