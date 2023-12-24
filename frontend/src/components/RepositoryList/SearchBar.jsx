
import { searchRepositories } from '../../utils';
import React, { useState, useEffect } from 'react';
import Autosuggest from 'react-autosuggest';
import { Search } from 'feather-icons-react';
import "bootstrap/dist/css/bootstrap.min.css";

// Define your searchRepositories function here...

const SearchBar = () => {
  const [inputValue, setInputValue] = useState('');
  const [suggestions, setSuggestions] = useState([]);


  const repos = [
    {
      author: 'C1',
      year: 1972
    },
    {
        author: 'C2',
      year: 2012
    },
    {
        author: 'C3',
        year: 2013
      },
      {
        author: 'Go',
        year: 2018
      },
  ];

    // Teach Autosuggest how to calculate suggestions for any given input value.
    const filterSuggestions = (value, results) => {
        const inputValue = value.trim().toLowerCase();
        const inputLength = inputValue.length;
        if(inputLength === 0){
            return []
        }else{
            return results.map((result) => {
                return {
                    author: result.full_name.split("/")[0],
                    repo: result.full_name.split("/")[1]
                }
            })
        }
    };


  const onSuggestionsFetchRequested = async ({ value }) => {
    // Fetch suggestions based on the current input value
    const results = await searchRepositories(value);
    let test = filterSuggestions(value, results)
    setSuggestions(test);
  };

  const onSuggestionsClearRequested = () => {
    // Clear suggestions when the input is cleared
    setSuggestions([]);
  };


  const renderSuggestion=(suggestion)=>(
    <div className='suggestion' onClick={()=>setInputValue(suggestion)}>
      {`${suggestion.author} / ${suggestion.repo}`}
    </div>
  );



  const eventEnter=(input)=>{
    searchRepositories(input);
    if(input.key == "Enter"){
      var split = input.target.value.split('/');
      var repo ={
        author: split[0].trim(),
        repo: split[1].trim(),
      };
      setSuggestions(repo);
    }
}

  return (
    <div className="autosuggest">

    <Autosuggest
      suggestions={suggestions}
      onSuggestionsFetchRequested={onSuggestionsFetchRequested}
      onSuggestionsClearRequested={onSuggestionsClearRequested}
      getSuggestionValue={suggestion => suggestion.name}
      renderSuggestion={renderSuggestion}
      onSuggestionSelected={eventEnter}
      inputProps={{
        placeholder: 'Search for repositories',
        value: inputValue,
        onChange: (_, { newValue }) => setInputValue(newValue),
      }}
    />
    <br />
     <button className='btn btn-primary' onClick={()=>console.log(inputValue)}>Checar Estado</button>

    </div>
  );
};

export default SearchBar;
