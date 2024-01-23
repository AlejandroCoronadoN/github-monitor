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
   * State that allows the user to test the application without any credentials
   *
   * @type {bool} Index of the hovered repository.
   */
  const [demo, setDemo] = useState(1); //Change to 1 start loading the demo

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

  const handleButtonClick = () => {
    if(demo>0){
        setDemo(0);
    }else{
        setDemo(1);

    }
  };

  /**
   * Main rendering of the App component.
   *
   * @returns {JSX.Element} The rendered App component.
   */
  return (
    <>
        <button className="demo-button"onClick={handleButtonClick}>
            Toggle Demo: {demo !==0? "ON" : "OFF"}
        </button>
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
          demo={demo}
          setDemo={setDemo}
        />
      </div>
    </>
  );
}

export default App;
