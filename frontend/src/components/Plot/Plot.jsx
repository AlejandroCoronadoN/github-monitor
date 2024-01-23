import "./Plot.css";
import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import Chart from 'chart.js/auto';

/**
 * Component representing a line chart to visualize weekly commits.
 *
 * @component
 * @param {Object} props - The component props.
 * @param {Array} props.plotsSeries - The data for the line chart, containing weekly commit information.
 * @param {number} props.hoverIndex - The index of the repository being hovered.
 * @returns {JSX.Element} The rendered Plot component.
 */


const Plot = ({ plotsSeries, hoverIndex, forecasted }) => {
  // State to store the maximum and minimum values for the y-axis
  const [maxY, setMaxY] = useState(3);
  const [minY, setMinY] = useState(-1);
  const [chartsDatasets, setChartsDatasets] = useState([]);
  const [chartLabels, setChartLabels] = useState([]);

  useEffect(() => {
    if (plotsSeries.length === 0) {
      setMaxY(5);
    } else {
      const maxYValue = Math.max(
        ...plotsSeries.flatMap((series) =>
          series.datasets[0].data.slice(0, 19),
        ),
      );
      const minYValue = Math.min(
        ...plotsSeries.flatMap((series) =>
          series.datasets[0].data.slice(0, 19),
        ),
      );
      const yScaleRange = maxYValue - minYValue;
      const yMax = maxYValue + yScaleRange * offsetMax;
      if (yMax === 0) {
        setMaxY(5);
      } else {
        setMaxY(yMax);
      }
    }
    // allwasy -1, leaves spaces between plots
    setMinY(-1);

    var data = plotsSeries.map((plotData, index) => {
      let backgroundColor = "#000000";

      // Set background color based on hover status and index
      if (hoverIndex === 1000) {
        backgroundColor = colors[index];
      } else {
        backgroundColor = colorsHover[index];
      }

      // Define the dataset for the current repository
      return {
        label: "",
        data: plotData.datasets[0].data.slice(-19),
        backgroundColor: "#ffffff",
        borderColor: backgroundColor,
        borderWidth: 3,
        hoverBorderWidth: 10,
        pointRadius: 8,
        pointHoverRadius: 15,
        pointHoverBorderColor: "rgba(0,0,0,0.2)",
        drawActiveElementsOnTop: false,
      };
    });

    var labels = [];
    if (plotsSeries.length !== 0) {
      labels = plotsSeries[0].labels.slice(-19);
    }

    setChartsDatasets(data);
    setChartLabels(labels); // Only uses first one
  }, [plotsSeries, hoverIndex]);

  // Create a unique color for each repository
  const colors = ["#4CCA8D", "#D65C5C", "#71B7F8"];
  const colorsHover = ["#b4ffdb", "#eaa3a3", "#c2dcf6"];

  // Calculate dynamic max and min values for y-axis
  const offsetMax = 0.01;

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const options = { month: "short", day: "numeric", year: "numeric" };
    return new Intl.DateTimeFormat("en-US", options).format(date);
  };

  // Tooltip handling for the line chart
  const getOrCreateTooltip = (chart) => {
    let tooltipEl = chart.canvas.parentNode.querySelector("div");

    if (!tooltipEl) {
      tooltipEl = document.createElement("div");
      tooltipEl.style.background = "#ffffff";
      tooltipEl.style.color = "#000000";
      tooltipEl.style.opacity = 1;
      tooltipEl.style.pointerEvents = "none";
      tooltipEl.style.position = "absolute";
      tooltipEl.style.transform = "translate(-50%, 0)";
      tooltipEl.style.transition = "all .1s ease";
      tooltipEl.style.boxShadow = "0px 2px 12px rgba(0,0,0,0.08)";
      tooltipEl.style.opacity = "0.1"; // Adjust the value (0 to 1) based on your preference

      const table = document.createElement("table");
      table.style.margin = "0px";

      tooltipEl.appendChild(table);
      chart.canvas.parentNode.appendChild(tooltipEl);
    }

    return tooltipEl;
  };

  // Custom external tooltip handler for the line chart
  const externalTooltipHandler = (context) => {
    const { chart, tooltip } = context;
    const tooltipEl = getOrCreateTooltip(chart);

    // Hide if no tooltip
    if (tooltip.opacity === 0) {
      tooltipEl.style.opacity = 0;
      return;
    }

    // Populate the tooltip content
    if (tooltip.body) {
      const titleLines = tooltip.title || [];
      const bodyLines = tooltip.body.map((b) => b.lines);

      const tableHead = document.createElement("thead");

      // Format and add title lines to the tooltip
      titleLines.forEach((title) => {
        const date = new Date(title);
        const options = {
          weekday: "short",
          year: "numeric",
          month: "short",
          day: "numeric",
        };

        const formattedDate = new Intl.DateTimeFormat("en-US", options).format(
          date,
        );
        const weekOfDate = `Week of ${formattedDate}`;

        const tr = document.createElement("tr");
        tr.style.borderWidth = 0;

        const th = document.createElement("th");
        th.style.borderWidth = 0;
        th.style.fontFamily = "Roboto";
        th.style.fontSize = "14px";
        th.style.fontWeight = 400;
        th.style.lineHeight = "16px";
        th.style.letterSpacing = "0em";
        th.style.textAlign = "center";
        th.style.color = "#6D6D90";

        const text = document.createTextNode(weekOfDate);

        th.appendChild(text);
        tr.appendChild(th);
        tableHead.appendChild(tr);
      });

      const tableBody = document.createElement("tbody");

      // Format and add body lines to the tooltip
      bodyLines.forEach((body, i) => {
        const colors = tooltip.labelColors[i];

        const span = document.createElement("span");
        span.style.background = "#ffffff";
        span.style.borderColor = "#ffffff";
        span.style.borderWidth = "2px";
        span.style.marginRight = "10px";
        span.style.height = "10px";
        span.style.width = "10px";
        span.style.display = "inline-block";

        const tr = document.createElement("tr");
        tr.style.backgroundColor = "inherit";
        tr.style.borderWidth = 0;

        const td = document.createElement("td");
        td.style.borderWidth = 0;
        td.style.fontFamily = "Roboto";
        td.style.fontSize = "14px";
        td.style.fontWeight = 700;
        td.style.lineHeight = "16px";
        td.style.letterSpacing = "0em";
        td.style.textAlign = "center";

        let nBody = Number(body);
        let commitText = "";

        // Customize the text based on the number of commits
        if (nBody === 1) {
          commitText = "\u29B5 " + body + " Commit";
        } else {
          commitText = "\u29B5 " + body + " Commits";
        }

        const text = document.createTextNode(commitText);

        td.appendChild(text);
        tr.appendChild(td);
        tableBody.appendChild(tr);
      });

      const tableRoot = tooltipEl.querySelector("table");

      // Remove old children
      while (tableRoot.firstChild) {
        tableRoot.firstChild.remove();
      }

      // Add new children
      tableRoot.appendChild(tableHead);
      tableRoot.appendChild(tableBody);
    }

    // Position and style the tooltip
    const { offsetLeft: positionX, offsetTop: positionY } = chart.canvas;
    tooltipEl.style.opacity = 1;
    tooltipEl.style.left = 115 + positionX + tooltip.caretX + "px";
    tooltipEl.style.top = -20 + positionY + tooltip.caretY + "px";
    tooltipEl.style.font = tooltip.options.bodyFont.string;
    tooltipEl.style.padding =
      tooltip.options.padding + "px " + tooltip.options.padding + "px";
  };

  // Chart options
  const options = {
    responsive: true,
    maintainAspectRatio: true,
    legend: {
      display: false,
    },
    scales: {
      x: {
        offset: true,
        borderCapStyle: "round",
        padding: 3,
        border: {
          color: "#37374A",
          width: 3,
        },
        ticks: {
          display: false,
        },
        grid: {
          display: true,
          tickLength: -10,
          tickWidth: 3,
          drawOnChartArea: false,
          tickColor: "#37374A",
        },
      },
      y: {
        //max: maxY,
        borderCapStyle: "round",
        min: minY,
        border: {
          color: "#37374A",
          width: 3,
        },
        ticks: {
          display: false,
        },
        grid: {
          display: true,
          tickLength: -10,
          tickWidth: 3,
          drawOnChartArea: false,
          tickColor: "#37374A",
        },
      },
    },
    elements: {
      line: {
        tension: 0.4,
      },
    },
    plugins: {
      tooltip: {
        enabled: false,
        position: "nearest",
        external: externalTooltipHandler,
      },
      legend: {
        display: false,
      },


    },
  };

  return (
    <div className="plot-graph-container">
      {/* Plot */}
      <div className="plot-graph">
        {/* Conditional rendering of the line chart based on the availability of repository data */}
        {plotsSeries.length === 0 ? (
          <div></div>
        ) : (
          <Line
            width={854}
            height={786}
            data={{ labels: chartLabels, datasets: chartsDatasets }}
            options={options}
          />
        )}
      </div>
    </div>
  );
};

export default Plot;
