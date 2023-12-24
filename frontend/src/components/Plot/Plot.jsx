/**
 * React Component: Plot
 * Description: This component utilizes React charts to create historic commits plots at a weekly level.
 * It receives repository data as props and renders a line chart using the React-chartjs-2 library.
 * @module Plot
 * @param {Object} repoData - Formatted data for GitHub repository commits to be displayed on the chart.
 * @returns {JSX.Element} - React component for rendering historic commits plots.
 */

import './style.css';
import { useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS } from "chart.js/auto";

const Plot = ({ repoData }) => {
  /**
   * Effect hook to log a message when the repository data is updated.
   * @function useEffect
   * @param {function} callback - Callback function to execute when dependencies change.
   * @param {Array} dependencies - Dependencies that trigger the effect when changed.
   * @returns {void}
   */
  useEffect(() => {
    console.log("RepoData UPDATED");
  }, [repoData]);

  return (
    <div className='plot-graph'>
      {/* Plot */}
      <div style={{ width: 500 }}>
        {/* Conditional rendering of the line chart based on the availability of repository data */}
        {repoData.length === 0 ? (
          <div></div>
        ) : (
          <Line data={repoData} />
        )}
      </div>
    </div>
  );
};

export default Plot;
