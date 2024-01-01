
import { searchRepositories } from '../../utils';
import { useState, useEffect } from 'react';
import { Search} from 'react-feather';
import "bootstrap/dist/css/bootstrap.min.css";
import './RepositoryList.css';
import { getCommits } from "../../utils";
import SelectedRepos from './ItemIterator/SelectedRepos';
import ItemDetails from './ItemIterator/ItemDetails';
import SelectionDetails from './ItemIterator/selectionDetails';
// Define your searchRepositories function here...

const RepositoryList = ({
    setSelectedRepositories,
    selectedRepositories,
    setPlotsSeries,
  }) => {
  const [inputValue, setInputValue] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [chartRepo, setChartRepo] = useState({})
  const localTest = false;
  let timeoutId;
    useEffect(() => {
      }, [suggestions]); // Trigger the effect whenever the input value changes

    const filterSuggestions = (value, results) => {
        const inputValue = value.trim().toLowerCase();
        const inputLength = inputValue.length;

        if(inputLength === 0){
            return []
        }else{
            let filteredValues = results.map((result, i) => {
              let lastUpdate = formatDate(result.updated_at);
                let formatedResponse =  {
                    id:i,
                    author: result.full_name.split("/")[0],
                    repository: result.full_name.split("/")[1],

                    stars: result.stargazers_count,
                    update: lastUpdate,
                }
                return formatedResponse
            })
            return filteredValues;
        }
      };

    const formatDate = (lastCommitTimestamp) => {
      const lastCommitDate = new Date(lastCommitTimestamp);
      // Get the current date and time
      const currentDate = new Date();

      // Calculate the time difference in milliseconds
      const timeDifference = currentDate - lastCommitDate;

      // Calculate the difference in hours
      const hoursDifference = timeDifference / (1000 * 60 * 60);

      if (hoursDifference < 24) {
        // If less than 24 hours, return the number of hours as an integer
        return "Updated" +  Math.floor(hoursDifference) + " hours ago";
      } else {
        // If more than 24 hours, format the original date
        const formattedDate = lastCommitDate.toLocaleDateString("en-US", {
          year: "numeric",
          month: "short",
          day: "numeric"
        });

      return `Updated on ${formattedDate}`;
      }
    }


    const fetchPlotsSeries = async () => {
      try {
        let allCommits = [];
        for (const item of selectedRepositories) {
          try {
            const weeklyCommits = await getCommits(item.author, item.repository);
            allCommits.push(weeklyCommits);
          } catch (error) {
            console.error(`Error fetching commits for ${item.author}/${item.repository}:`, error.message);
            // You might want to handle the error here or throw it again
            throw error;
          }
        }

        // Assuming setPlotsSeries expects an array of commits data
        setPlotsSeries(allCommits);
        console.log("ALL COMMITS: ", allCommits)
      } catch (error) {
        console.error('Error fetching commits:', error.message);
        // Handle errors (e.g., display an error message to the user)
      }
    };
    const [requestTimeOut, setrequestTimeOut] = useState(false);
    const [startTimeOut, setStartTimeOut] = useState(false);

    const preHandleChange = (event) =>{
      const currentTime = new Date();
      if(startTimeOut){
        // Calculate the time difference in milliseconds
        const timeDifferenceInMilliseconds = requestTimeOut.getTime() - currentTime.getTime();
        const diff = timeDifferenceInMilliseconds / 1000;
        if(diff>1 ){
          handleChange(event)
          setrequestTimeOut(currentTime)
        }else{
          setrequestTimeOut(currentTime)
        }
      }else{
        setStartTimeOut(true)
        setrequestTimeOut(currentTime)

      }


    }

    const handleChange = (event) => {
      const inputValue = event.target.value;
      setInputValue(inputValue);

      // Clear the previous timeout
      clearTimeout(timeoutId);

      // Set a new timeout for 1000 milliseconds (1 second)
      timeoutId = setTimeout(async () => {
        const results = await searchRepositories(inputValue);
        let filteredSuggestions = filterSuggestions(inputValue, results);
        setSuggestions(filteredSuggestions);
      }, 1000); // Adjust the delay to 1000 milliseconds (1 second)

    };

    const handleSuggestionClick = async(item) => {
      setSelectedRepositories([...selectedRepositories, item]);
      //filterItem Data
      const test = await fetchPlotsSeries()
      console.log(`test: ${test}`);
    };

    const handleSelectionClicked = (item) => {
      //This function tells the plot component wich plot should be highlighted.
      setChartRepo(item);
      //filterItem Data
    };

  return (
    <div className="repository-list-container">
    <input className="item-list-header" onChange={handleChange}/>
    <div className="item-list">
      {suggestions.map((item, id) => {
        return (
        <div>
            <ItemDetails
              key={id}
              id={id}
              item={item}
              handleSuggestionClick ={handleSuggestionClick}
            />
        </div>
        )
      })}
    </div>

      {suggestions.length !==0?
        <div>
          <br></br>
        </div>
        :
        <div className="state-message">
          <div className="search-icon">
            <Search size={49} style={{  display: 'block' }} />

          </div>
          <span style={{ display: 'block' }}>Search for a Github repository to populate the graph</span>
        </div>

      }

    <div className="item-list">
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
  </div>

  );
};

export default RepositoryList;
