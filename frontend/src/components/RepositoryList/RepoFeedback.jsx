import React, {useEffect} from 'react';
import { fetchLastIssue } from '../../utils';

const RepoFeddback = ({
    sentimentCategory,
    description }) => {

    useEffect(() => {
        // Replace 'your_username' and 'your_repository' with the actual GitHub username and repository name
        const username = 'your_username';
        const repo = 'your_repository';

        // Initialize Octokit with your GitHub personal access token
        const octokit = new Octokit({
            auth: 'your_personal_access_token',
        });

    // Call the fetchLastIssue function
    fetchLastIssue();
  }, []); // Dependency array to run the effect only once


  const getEmoji = (intensity) => {
    switch (intensity) {
      case 'fatal':
        return 'ð¥'; // Replace with your preferred emoji
      case 'important':
        return 'â'; // Replace with your preferred emoji
      case 'conflicts':
        return 'ð¤¯'; // Replace with your preferred emoji
      case 'noProblems':
        return 'â'; // Replace with your preferred emoji
      case 'perfect':
        return 'ð'; // Replace with your preferred emoji
      default:
        return '';
    }
  };

  return (
    <div>
      <h2>Problem Intensity:</h2>
      <p>Loading...</p>
      <p>{getEmoji(sentimentCategory)}</p>
      <p className={author} >{description}</p>
    </div>
  );
};

export default RepoFeddback;
