/**
 * React Component: RepositoryList
 * Description: This component allows the user to search for GitHub repositories,
 * retrieve commit data, and pass the formatted data to the Plot component for rendering
 * a GitHub commit history chart per repository.
 * @module RepositoryList
 * @param {function} setRepoData - Function to set the repository data for the Plot component.
 * @returns {JSX.Element} - React component for the repository list.
 */

import { useState } from 'react';
import './style.css';
import SearchBar from './SearchBar';
import { getAllCommits, weeklyCommits, formatChatData } from "../../utils";

const RepositoryList = ({ setRepoData }) => {
  // State to store the commits data for the selected repository
  const [commits, setCommits] = useState("");

  /**
   * Asynchronous function to fetch commits data for a specific GitHub repository,
   * format the data, and set it for the Plot component.
   * @async
   * @function getCommits
   * @returns {void}
   */
  const getCommits = async () => {
    if (commits === "") {
      try {
        // GitHub repository information (replace with user-specific values)
        let owner = 'AlejandroCoronadoN';
        let repo = 'tellatale';

        // Fetch all commits for the specified repository
        let commitsData = await getAllCommits(owner, repo);

        // Set the commits data to the state
        setCommits(commitsData);

        // Group commits into weekly intervals and calculate total commits for each week
        let plotData = weeklyCommits(commitsData);

        // Log the formatted plot data to the console (for debugging)
        console.log(`REPO LIST plot_data: ${plotData}`);

        // Format the commit data for chart presentation
        let formattedData = formatChatData(plotData);

        // Set the formatted repository data for the Plot component
        setRepoData(formattedData);
      } catch (error) {
        console.error('Error fetching commits:', error.message);
        // Handle errors (e.g., display an error message to the user)
      }
    }
  };

  return (
    <div className='repo-list'>
      {/* RepositoryList */}
      <div>
        {/* Button to fetch commits data for the selected repository */}
        <SearchBar/>
      </div>
    </div>
  );
};

export default RepositoryList;
