import React, { useState } from "react";
function ConditionalRender() {
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    return (
        <div>
            {isLoggedIn ? <h2>Welcome User</h2> : <h2>Please Login</h2>}
            <button onClick={() => setIsLoggedIn(!isLoggedIn)}>
                Toggle
            </button>
        </div>
    );
}
export default ConditionalRender; 