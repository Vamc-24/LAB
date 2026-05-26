import React from "react";
function App() {
    const subjects = ["DBMS", "AI", "ML", "React.js", "Python"];
    return (
        <div style={{ textAlign: "center", marginTop: "50px" }}>
            <h2>Iterative Rendering using map()</h2>
            <ul style={{ listStyleType: "none" }}>
                {subjects.map((sub, index) => (
                    <li key={index} style={{ fontSize: "20px" }}>
                        {index + 1}. {sub}
                    </li>
                ))}

            </ul>
        </div>
    );
}
export default App; 