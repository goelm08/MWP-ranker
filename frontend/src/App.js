import {useEffect, useState} from "react";
import * as React from "react";
import TextField from "@mui/material/TextField";
import List from "./Components/List";
import "./App.css";
import * as events from "events";


function App() {
    const [inputText, setInputText] = useState([]);
    const [message, setMessage] = useState("");
    const [eqn, setEqn] = useState("");

    const handleChange = (event) => {
        setMessage(event.target.value);
    };

    const changeEqn = (event) => {
        setEqn(event.target.value);
    }

    let inputHandler = (e) => {
        //convert input text to lower case
        console.log(message)
        // const lowerCase = message.toLowerCase();
        const params = {
            question: message,
        };
        const options = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            json:true,
            body: JSON.stringify( params )
        };

        fetch("/solve",options).then((res) =>
            res.json().then((data) => {
                const ans_list = data['list']
                setInputText(ans_list);
                setEqn(data['eqn'])
                console.log(data);
            }).catch(error=>{
                //if debug
                //throw(error)
                const randomArray = (length, max) => [...new Array(length)]
                    .map(() => Math.round(Math.random() * max));
                setInputText(randomArray(50, 2875))
            })
        );

    };

    return (
        <div className="main">
            <h1>MWP solver & retreiver</h1>
            <div className="search">
                <TextField
                    id="outlined-basic"
                    variant="outlined"
                    onChange={handleChange}
                    fullWidth
                    label="Search"
                />

            </div>
            <button onClick={inputHandler} >
                Search
            </button>
            <div>
                <TextField
                    id="outlined-eqn"
                    variant="outlined"
                    value={eqn}
                    fullWidth
                />
            </div>
            <List input={inputText} />
        </div>
    );
}

export default App;