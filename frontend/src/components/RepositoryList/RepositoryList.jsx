import { searchRepositories, weeklyCommits } from '../../utils';
import React, { useState, useEffect } from 'react';
import { Search } from 'react-feather';
import "bootstrap/dist/css/bootstrap.min.css";
import './RepositoryList.css';
import { getRepoInfo, getRepoForecast } from "../../utils";
import ItemDetails from './ItemIterator/ItemDetails';
import SelectionDetails from './ItemIterator/SelectionDetails';
import repoSuggestions from './repositorySearch.json';
import RepoFeddback from './RepoFeedback';

/**
 * RepositoryList component displays a list of GitHub repositories with search functionality.
 *
 * @param {Object} props - Component props.
 * @param {Function} props.setSelectedRepositories - Function to set selected repositories.
 * @param {Array} props.selectedRepositories - Array of selected repositories.
 * @param {Function} props.setPlotsSeries - Function to set plot series data.
 * @param {Function} props.setHoverIndex - Function to set the hover index.
 * @returns {JSX.Element} React component.
 */
const RepositoryList = ({
  setSelectedRepositories,
  selectedRepositories,
  setPlotsSeries,
  plotsSeries,
  setHoverIndex,
}) => {
  const [inputValue, setInputValue] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [chartRepo, setChartRepo] = useState({});
  const [colorIndex, setColorIndex] = useState(0);
  const [loading, setLoading] = useState(false); // New loading state
  const [requestTimeOut, setrequestTimeOut] = useState(false);
  const [startTimeOut, setStartTimeOut] = useState(false);

    /**
     * State hook to manage description for the Loading component
     *
     * @type {string} Index of the hovered repository.
     */
    const [description, setDescription] = useState("");

    /**
     * State hook to manage sentimentAnalysis category and insert Emoji
     *
     * @type {string} Index of the hovered repository.
     */
    const [sentimentCategory, setSentimentCategory] = useState("");



  const colors = ['#4CCA8D', '#71B7F8', '#D65C5C'];
  const localTest = false;
  let timeoutId;

  useEffect(() => {
    // Trigger the effect whenever the input value changes
  }, [suggestions, description, sentimentCategory]);

  /**
   * Filters the GitHub repository suggestions based on the input value.
   *
   * @param {string} value - Input value.
   * @param {Array} results - Array of repository search results.
   * @returns {Array} Filtered suggestion values.
   */
  const filterSuggestions = (value, results) => {
    const inputValue = value.trim().toLowerCase();
    const inputLength = inputValue.length;

    if (inputLength === 0) {
      return [];
    } else {
      let filteredValues = results.map((result, i) => {
        let author = result.full_name.split("/")[0];
        let repo = result.full_name.split("/")[1];

        if (author.length >= 20) {
          author = author.slice(0, 40) + "...";
        }
        if (repo.length > 20) {
          repo = repo.slice(0, 30) + "...";
        }

        let lastUpdate = formatDate(result.updated_at);
        let formatedResponse = {
          id: i,
          author: author,
          repository: repo,
          stars: result.stargazers_count,
          update: lastUpdate,
        };
        return formatedResponse;
      });

      return filteredValues;
    }
  };

  /**
   * Formats the last commit timestamp into a user-friendly update message.
   *
   * @param {string} lastCommitTimestamp - Last commit timestamp.
   * @returns {string} Formatted update message.
   */
  const formatDate = (lastCommitTimestamp) => {
    const lastCommitDate = new Date(lastCommitTimestamp);
    const currentDate = new Date();

    const timeDifference = currentDate - lastCommitDate;
    const hoursDifference = timeDifference / (1000 * 60 * 60);

    if (hoursDifference < 24) {
      return "Updated " + Math.floor(hoursDifference) + " hours ago";
    } else {
      const formattedDate = lastCommitDate.toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      });

      return `Updated on ${formattedDate}`;
    }
  };

  /**
   * Fetches plot series data for selected repositories.
   *
   * @param {Array} newSelectedRepos - Array of newly selected repositories.
   * @returns {Promise<Array>} Array of plot series data.
   */

  const fetchPlotsSeries = async (item) => {
    try {
        setLoading(true); // Set loading to true before starting the fetch
      const infoResponse = await getRepoInfo(item.author, item.repository)

      setDescription(infoResponse.llmDescription)
      setSentimentCategory(infoResponse.sentimentCategory)
      setLoading(true); // Set loading to true before starting the fetch

      const response = await getRepoForecast(item.author, item.repository)

      setLoading(false); // Set loading to false after the fetch is complete
      return response;
    } catch (error) {
      setLoading(false); // Set loading to false if an error occurs
      console.error(`Error fetching commits for ${item.author}/${item.repository}:`, error.message);
      throw error;
    }
  };

  /**
   * Handles pre-change events before updating the input value.
   *
   * @param {Object} event - Input change event.
   */
  const preHandleChange = (event) => {
    const currentTime = new Date();
    if (startTimeOut) {
      const timeDifferenceInMilliseconds = requestTimeOut.getTime() - currentTime.getTime();
      const diff = timeDifferenceInMilliseconds / 1000;
      if (diff > 1) {
        handleChange(event);
        setrequestTimeOut(currentTime);
      } else {
        setrequestTimeOut(currentTime);
      }
    } else {
      setStartTimeOut(true);
      setrequestTimeOut(currentTime);
    }
  };

  /**
   * Handles the input change event and triggers repository search.
   *
   * @param {Object} event - Input change event.
   */
  const handleChange = (event) => {
    const inputValue = event.target.value;
    setInputValue(inputValue);

    clearTimeout(timeoutId);

    timeoutId = setTimeout(async () => {
      let test = false;
      let results = [];

      if (test) {
        results = repoSuggestions;
      } else {
        results = await searchRepositories(inputValue);
      }

      let filteredSuggestions = filterSuggestions(inputValue, results);
      setSuggestions(filteredSuggestions);
    }, 10); // Adjust the delay to 1000 milliseconds (1 second)
  };

  /**
   * Handles click events on suggestion items and updates selected repositories.
   *
   * @param {Object} item - Selected suggestion item.
   */
  const handleSuggestionClick = async (item) => {
    setSuggestions([]);
    let newSelectedRepos = [...selectedRepositories, item];
    setSelectedRepositories(newSelectedRepos);
    let newCommits = await fetchPlotsSeries(item);



    // Check if plotsSeries already has 3 elements
    if (plotsSeries.length === 3) {
        // Remove the first element
        let updatedPlotsSeries = plotsSeries.slice(1);
        // Update the ids for the remaining elements
        updatedPlotsSeries = updatedPlotsSeries.map((elem) => ({ ...elem, id: elem.id - 1 }));
        // Add the new element with id 2
        updatedPlotsSeries.push({ ...newCommits, id: 2 });

        setPlotsSeries(updatedPlotsSeries);
    } else {
        // Add the new element with the next id
        let newPlotsSeries = [...plotsSeries, { ...newCommits, id: plotsSeries.length }];
        setPlotsSeries(newPlotsSeries);
    }

  };

  /**
   * Handles click events on selected repositories and updates chart highlight.
   *
   * @param {Object} item - Selected repository item.
   */
  const handleSelectionClicked = (item) => {
    setHoverIndex(item.id);
  };

  return (
    <div className="repository-list-container">
        <div className="item-list-header">
            <input className="item-list-header-input" onChange={handleChange} />
            <div className="item-list-icon">
                <Search size={22.5} style={{ display: 'block' }} />
            </div>

        </div>


      {loading && (
        <div className="loading-alert">
          <div className="loading-message">
            <RepoFeddback
                description ={description}
                sentimentCategory = {sentimentCategory}
            />
          </div>
        </div>
      )}

      <div className="item-list-suggested">
        {suggestions.map((item, id) => (
          <div className="item-suggestion" key={id}>
            <ItemDetails
              id={id}
              item={item}
              handleSuggestionClick={handleSuggestionClick}
            />
          </div>
        ))}
      </div>

      {(suggestions.length !== 0) | (selectedRepositories.length !== 0) ? (
        <div>
          <br></br>
        </div>
      ) : (
        <div className="state-message">
          <div className="search-icon">
            <Search size={49} style={{ display: 'block' }} />
          </div>
          <span className='state-text'>Search for a Github repository to populate the graph</span>
        </div>
      )}

      <div className="item-list-selected">
        {selectedRepositories.map((item, id) => (
          <div key={id}>
            <SelectionDetails
              id={id}
              item={item}
              handleSelectionClicked={handleSelectionClicked}
              selectedColor={colors[id]}
              setPlotsSeries={setPlotsSeries}
              plotsSeries = {plotsSeries}
              setHoverIndex={setHoverIndex}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default RepositoryList;
