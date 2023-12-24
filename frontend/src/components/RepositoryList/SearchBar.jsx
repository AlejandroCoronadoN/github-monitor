import {useState} from 'react'
import { Search } from 'feather-icons-react';
import { searchRepositories } from '../../utils';
const SearchBar = () => {

    const searchRepo = async () => {
        let repos = searchRepositories(inputValue)
        let answ = JSON.stringify(repos)
        console.log(`*** repos: ${answ}`);
    }
    const [inputValue, setInputValue] = useState("");

  return (
    <div>
    <input
    className="textbox disclaimer"
    type="text"
    placeholder={""}
    alt="self-checkout"
    onChange={(e) => setInputValue(e.target.value)}
    value={inputValue}
    />

    <button className="login-button" onClick={searchRepo}>
            <Search/>
    </button>
    </div>

)}

export default SearchBar
