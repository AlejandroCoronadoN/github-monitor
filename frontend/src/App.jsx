import './App.css';
import React, { useState } from 'react';
import Plot from './components/Plot/Plot';
import RepositoryList from './components/RepositoryList/RepositoryList';
import jsonData from './data.json'; // Adjust the path as needed

/**
 * Main application component.
 *
 * @component
 * @returns {JSX.Element} The rendered App component.
 */
function App() {
  /**
   * State hook to manage selected repositories.
   *
   * @type {Array} Array of selected repositories.
   */
  const [selectedRepositories, setSelectedRepositories] = useState([]);

  /**
   * State hook to manage data series for plots.
   *
   * @type {Array} Array of data series for plots.
   */
  const [plotsSeries, setPlotsSeries] = useState([]);

  /**
   * State hook to manage hover index for plots.
   *
   * @type {number} Index of the hovered repository.
   */
  const [hoverIndex, setHoverIndex] = useState(1000);



  /**
   * Main rendering of the App component.
   *
   * @returns {JSX.Element} The rendered App component.
   */
  return (
    <>
      <div className="container">
        {/* Plot component displaying data series */}
        <Plot
          plotsSeries={plotsSeries}
          hoverIndex={hoverIndex}
        />

        {/* RepositoryList component for selecting repositories */}
        <RepositoryList
          setSelectedRepositories={setSelectedRepositories}
          selectedRepositories={selectedRepositories}
          setPlotsSeries={setPlotsSeries}
          plotsSeries = {plotsSeries}
          setHoverIndex={setHoverIndex}

        />
      </div>
    </>
  );
}

export default App;
