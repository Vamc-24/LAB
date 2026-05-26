import React, { useState } from "react";
function App() {
    const [count, setCount] = useState(0);
    return (
        <div style={{ textAlign: "center", marginTop: "80px" }}>
            <h2>Counter using useState</h2>
            <h1>{count}</h1>
            <button onClick={() => setCount(count + 1)}>Increment</button>
            <button onClick={() => setCount(count - 1)} style={{ marginLeft: "10px" }}>
                Decrement
            </button>
            <button onClick={() => setCount(0)} style={{ marginLeft: "10px" }}>
                Reset
            </button>
        </div>
    );
}
export default App; 