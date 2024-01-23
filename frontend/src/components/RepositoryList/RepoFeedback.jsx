import React, { useEffect, useState } from "react";

const RepoFeddback = ({ 
    sentimentCategories, 
    descriptions, 
    selectedRepositories, 
    loading }) => {
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    let intervalId;
    let maxindex =selectedRepositories.length

    const updateIndex = () => {
        
      setCurrentIndex((prevIndex) => (prevIndex + 1) % maxindex);
    };

    if (loading) {
      // Start updating index every 5 seconds
      intervalId = setInterval(updateIndex, 5000);
    }

    return () => {
      // Clear the interval on component unmount
      clearInterval(intervalId);
    };
  }, [loading]);

  const getEmoji = (intensity) => {
    switch (intensity) {
      case "fatal":
        return "💀"; // Replace with your preferred emoji
      case "important":
        return "⚠️"; // Replace with your preferred emoji
      case "conflicts":
        return "👁️"; // Replace with your preferred emoji
      case "noproblems":
        return "✅"; // Replace with your preferred emoji
      case "perfect":
        return "✨"; // Replace with your preferred emoji
      default:
        return "";
    }
  };

  return (
    <div>
      <p className="feed-title">Processing forecast...</p>
      {selectedRepositories[currentIndex] ? (
        <p className="feed-subtitle">{selectedRepositories[currentIndex].author + "/" +selectedRepositories[currentIndex].repo}</p>
        ) : (
        <p className="feed-subtitle">Author Not Available</p>
        )}

      <p className="feed-message">{descriptions[currentIndex]}</p>
      <br></br>
      <span role="img" aria-label="sheep">
        {getEmoji(sentimentCategories[currentIndex])}
      </span>
      <br></br>
    </div>
  );
};

export default RepoFeddback;

