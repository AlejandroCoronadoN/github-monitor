import React, {useEffect} from 'react';
import { fetchLastIssue } from '../../utils';

const RepoFeddback = ({
    sentimentCategory,
    description }) => {

    useEffect(() => {


    }, [sentimentCategory]);
  const getEmoji = (intensity) => {
    switch (intensity) {
      case 'fatal':
        return "💀"; // Replace with your preferred emoji
      case 'important':
        return "⚠️"; // Replace with your preferred emoji
      case 'conflicts':
        return "👁️"; // Replace with your preferred emoji
      case 'noproblems':
        return "✅"; // Replace with your preferred emoji
      case 'perfect':
        return "✨"; // Replace with your preferred emoji
      default:
        return '';
    }
  };

  return (
    <div>
      <p>Loading...</p>
      <br></br>
      <span role="img" aria-label="sheep">
        {getEmoji(sentimentCategory)}</span>
        <br></br>
      <p className="description-message" >{description}</p>
    </div>
  );
};

export default RepoFeddback;
