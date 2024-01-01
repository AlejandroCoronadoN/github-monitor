import './Plot.css';
import { useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS } from 'chart.js/auto';

const Plot = ({ plotsSeries }) => {
  useEffect(() => {
    console.log("RepoData UPDATED");
  }, [plotsSeries]);

  // Create a unique color for each repository
  const colors = ['#4CCA8D', '#D65C5C', '#71B7F8', '#4CCA8D', '#D65C5C', '#71B7F8','#4CCA8D', '#D65C5C', '#71B7F8',];

  // Point 1: Create separate datasets for each repository
  const chartDatasets = plotsSeries.map((plotData, index) => ({
    label: plotData.datasets[0].label.slice(-7),
    data: plotData.datasets[0].data.slice(-7),
    backgroundColor: colors[index],
    borderColor: 'black',
    borderWidth: 2,
  }));

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const options = { month: 'short', day: 'numeric', year: 'numeric' };
    return new Intl.DateTimeFormat('en-US', options).format(date);
  };

  const chartOptions = {
    plugins: {
      tooltip: {
        callbacks: {
          // Point 2: Show both date and commit history on hover
          label: ({ datasetIndex, dataIndex }) => {
            const date = formatDate(plotsSeries[datasetIndex].labels[dataIndex]);
            const commits = plotsSeries[datasetIndex].datasets[0].data[dataIndex];
            const weekOf = `Week of ${date}`;
            const commitsLabel = `**${commits} Commits**`;
            return `${weekOf}\n${commitsLabel}`;
          },
        },
      },
    },
    elements: {
      line: {
        tension: 0.2, // Point 2: Adjust tension for smoother lines
        borderWidth: 8, // Point 2: Increase line thickness
      },
      ticks: {
        color: 'blue',
        lineWidth: 12,
        stepSize: 1,
      },
      lineWidth: 12, // Adjust the thickness of the x-axis

    },
    scales: {
      x: {
        grid: {
          display: false,
        },
        ticks: {
          display: true,
        },
        lineWidth: 12, // Adjust the thickness of the x-axis

      },
      y: {
        beginAtZero: true,
        grid: {
          display: false,
        },

        ticks: {
          color: 'black',
          lineWidth: 12,
          stepSize: 1,
        },
      },
    },
    responsive: true,
    maintainAspectRatio: false,
  };

  return (
    <div className='plot-graph'>
      {/* Plot */}
      <div style={{  width: '94%', height: '87%',backgroundColor: 'white' }}>
        {/* Conditional rendering of the line chart based on the availability of repository data */}
        {plotsSeries.length === 0 ? (
          <div></div>
        ) : (
          <Line data={{ labels: plotsSeries[0].labels.slice(0,7), datasets: chartDatasets }} options={chartOptions} />
        )}
      </div>
    </div>
  );
};

export default Plot;
