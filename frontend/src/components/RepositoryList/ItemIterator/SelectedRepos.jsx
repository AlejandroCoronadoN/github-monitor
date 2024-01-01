
import { searchRepositories } from '../../../utils';
import { useState, useEffect } from 'react';
import { Search } from 'feather-icons-react';
import "bootstrap/dist/css/bootstrap.min.css";
import './ItemIterator.css';
import SelectionDetails from './selectionDetails';

// Define your searchRepositories function here...

const SelectedRepos = ({selectedRepositories}) => {
  const [clickedRepo, setClickedRepo] = useState([]);
  const localTest = false;
    // useEffect for handling suggestions
    useEffect(() => {
        console.log(`selectedRepositories ueh: ${selectedRepositories}`);
        }, [selectedRepositories]); // Trigger the effect whenever the input value changes

    const handleSelectionClicked = (item) => {
        setClickedRepo(item);
    };

  return (

    <div className="item-list ">
      {selectedRepositories.map((item, id) => {
        return (
        <div>
            <SelectionDetails
              key={id}
              id={id}
              item={item}
              handleSelectionClicked ={handleSelectionClicked}
            />
        </div>
        )
      })}
    </div>

  );
};

export default SelectedRepos;
