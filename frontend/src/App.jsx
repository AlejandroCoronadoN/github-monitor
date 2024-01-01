
import './App.css'
import { useState } from 'react';
import Plot from './components/Plot/Plot'
import RepositoryList from './components/RepositoryList/RepositoryList';
import SelectedRepos from './components/RepositoryList/ItemIterator/SelectedRepos';
import jsonData from './data.json'; // Adjust the path as needed

function App() {
    const [selectedRepositories, setSelectedRepositories] = useState([]);
    const [plotsSeries, setPlotsSeries] = useState(jsonData);
  return (
    <>

      <div className="container">
        <Plot
            plotsSeries = {plotsSeries}
        />
        <RepositoryList
            setSelectedRepositories={setSelectedRepositories}
            selectedRepositories={selectedRepositories}
            setPlotsSeries = {setPlotsSeries}
        />

      </div>



    </>
  )
}

export default App
