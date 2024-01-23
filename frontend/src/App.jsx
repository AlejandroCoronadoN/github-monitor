import React, { useState } from 'react';
import Plot from './components/Plot/Plot';
import RepositoryList from './components/RepositoryList/RepositoryList';

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
   * State that tells the plot component if a forecast has been created, if so component will create a vertical line that shows when the forecast starts.
   *
   * @type {bool} Index of the hovered repository.
   */
  const [forecasted, setForecasted] = useState(false)


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
          forecasted ={forecasted}
        />

        {/* RepositoryList component for selecting repositories */}
        <RepositoryList
          setSelectedRepositories={setSelectedRepositories}
          selectedRepositories={selectedRepositories}
          setPlotsSeries={setPlotsSeries}
          plotsSeries = {plotsSeries}
          setHoverIndex={setHoverIndex}
          setForecasted ={setForecasted}

        />
      </div>
    </>
  );
}

export default App;
