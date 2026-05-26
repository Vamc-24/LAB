import React from "react";
function Student(props) {
    return (
        <div>
            <h3>Student Name: {props.name}</h3>
            <h3>Branch: {props.branch}</h3>
        </div>
    );
}
export default Student; 