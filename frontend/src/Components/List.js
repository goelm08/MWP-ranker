import { React, useState } from 'react'
import data from "./ListData.json"

function List(props) {
    //create a new array by filtering the original array
    const filteredData = data.filter((el) => {
        // debugger;
        return props.input.includes(el.id);
        //if no input the return the original
        // if (props.input === '') {
        //     return el;
        // }
        // //return the item which contains the user input
        // else {
        //     return el.ques.toLowerCase().includes(props.input)
        // }
    })
    return (

        <table>
            <tr>
                <th>ID</th>
                <th>Question</th>
                <th>Equation</th>
            </tr>
            {filteredData.map((item, key) => {
                return (
                    <tr key={key}>
                        <td>{item.id}</td>
                        <td>{item.ques}</td>
                        <td>{item.eqn}</td>
                    </tr>
                )
            })}
        </table>
    )
}

export default List