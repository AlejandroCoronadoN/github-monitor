
import './App.css'
import { useState } from 'react';
import Plot from './components/Plot/Plot'
import RepositoryList from './components/RepositoryList/RepositoryList'
import { UserData } from "./components/Plot/Data";

function App() {
    const [repoData, setRepoData] = useState([]);

  return (
    <>

      <h1>Github Monito Tool App</h1>

      <div className="container">
        <Plot repoData={repoData}/>
        <RepositoryList setRepoData={setRepoData}/>
      </div>



    </>
  )
}

export default App
