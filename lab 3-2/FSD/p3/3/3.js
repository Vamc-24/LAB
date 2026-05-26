import React from "react";
import Student from "./Student";
function App() {
    const studentName = "Sai Kiran";
    const studentBranch = "CSD / AIML";
    return (
        <div style={{ textAlign: "center", marginTop: "50px" }}>
            <h2>Props Example</h2>
            <Student name={studentName} branch={studentBranch} />
        </div>
    );
}
export default App; 