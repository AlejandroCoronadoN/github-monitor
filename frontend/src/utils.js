/**
 * Module: GitHub API Utilities
 * Description: This module contains functions for interacting with the GitHub API,
 * retrieving commit data, and formatting it for chart presentation.
 * @module githubApiUtils
 */
import { Octokit } from "octokit";
import searchRepoDemo from "./demo/searchRepo.json";
import pytorchCommits from "./demo/allCommits_1.json";
import reactCommits from "./demo/allCommits_2.json";
import phpCommits from "./demo/allCommits_3.json";
import descriptionDemo from "./demo/fetchDescription.json";
import issueDemo from "./demo/fetchLastIssue.json";

//import commitsData from "./commitsData.json"
// GitHub API token for authentication
const githubApiToken = import.meta.env.VITE_GITHUB_API_TOKEN;
// Initialize Octokit instance with authentication
const octokit = new Octokit({
  auth: githubApiToken,
});
//const pytorchCommits = import.meta.glob("./demo/allCommits_1.json");
//const reactCommits = import.meta.glob("./demo/allCommits_2.json");

/**
 * Formats commit data for chart presentation.
 *
 * @param {Object[]} plotData - List of objects containing commit data.
 * @returns {Object} - Formatted data suitable for charting.
 */
export const formatChatData = (plotData) => {
  let formatChatData = {
    labels: plotData.map((data) => data.startDate),
    datasets: [
      {
        label: "Commit History",
        data: plotData.map((data) => data.totalCommits),
        backgroundColor: [
          "rgba(75,192,192,1)",
          "#ecf0f1",
          "#50AF95",
          "#f3ba2f",
          "#2a71d0",
        ],
        borderColor: "black",
        borderWidth: 2,
      },
    ],
  };
  return formatChatData;
};

/**
 * Formats forecast data for chart presentation.
 *
 * @param {Object[]} plotData - List of objects containing commit data.
 * @returns {Object} - Formatted data suitable for charting.
 */
export const getPlotFormat = (response, forecastSubprocess) => {
  let formattedDates = [];
  let data = [];
  if (forecastSubprocess) {
    formattedDates = response.dates.map((timestamp) => {
      const date = new Date(timestamp / 1e6); // Convert nanoseconds to milliseconds
      return date.toLocaleDateString("en-US"); // Adjust the locale as needed
    });
    data = response.forecast;
  } else {
    formattedDates = response.dates.map((date) => {
      return date.substring(0, 10); // Adjust the locale as needed
    });
    data = response.commits;
  }

  let formatChatData = {
    labels: formattedDates,
    datasets: [
      {
        label: "Commit History",
        data: data,
        backgroundColor: [
          "rgba(75,192,192,1)",
          "#ecf0f1",
          "#50AF95",
          "#f3ba2f",
          "#2a71d0",
        ],
        borderColor: "black",
        borderWidth: 2,
      },
    ],
  };
  return formatChatData;
};

/**
 * Groups commits into weekly intervals and calculates the total number of commits for each week.
 *
 * @param {Object[]} commits - List of commits to be grouped.
 * @returns {Object[]} - List of objects representing weekly commit data.
 */
export const weeklyCommits = (commitsData) => {
  // Initialize an array to store weekly commit lists
  const plotData = [];

  // Get the current date
  const currentDate = new Date();

  // Iterate for 10 weeks
  let n_weeks = 80;
  for (let week = 0; week < n_weeks; week++) {
    // Calculate the start and end dates of the current week window
    const endDate = new Date(currentDate - week * 7 * 24 * 60 * 60 * 1000);
    const startDate = new Date(endDate - 7 * 24 * 60 * 60 * 1000);

    // Filter commits within the current week window
    const weeklyCommits = commitsData.filter((commit) => {
      const commitDate = new Date(commit.commit.author.date);
      return commitDate >= startDate && commitDate <= endDate;
    });

    // Add the weekly commits to the list
    plotData.push({
      id: n_weeks - week,
      startDate: startDate.toISOString(),
      endDate: endDate.toISOString(),
      commits: weeklyCommits,
      totalCommits: weeklyCommits.length,
    });
  }
  plotData.sort((a, b) => a.id - b.id);
  return plotData;
};

/**
 * Fetches all commits for a given repository from the GitHub API.
 *
 * @param {string} owner - Owner of the GitHub repository.
 * @param {string} repo - Name of the GitHub repository.
 * @returns {Object[]} - List of all commits for the specified repository.
 * @throws {Error} - Throws an error if there is an issue fetching commits.
 */
export const getAllCommits = async (owner, repo) => {
  let i = 1;
  let allCommits = [];
  let dataContinue = true;
  const currentDate = new Date();
  //currentDate.setFullYear(currentDate.getFullYear() - 1);
  currentDate.setDate(currentDate.getDate() - 30);
  const formattedDate = currentDate.toISOString();

  try {
    while (dataContinue) {
      const commitsResponse = await octokit.request(
        "GET /repos/{owner}/{repo}/commits",
        {
          owner: owner,
          repo: repo,
          since: formattedDate,
          per_page: 100, // Adjust per_page as needed
          page: i,
          headers: {
            Accept: "application/vnd.github.v3+json", // Use the recommended Accept header
          },
        }
      );

      const commits = commitsResponse.data;

      if (commits.length === 0) {
        // No more commits, break out of the loop
        dataContinue = false;
      }

      // Append commits to the list
      allCommits = allCommits.concat(commits);

      // Increment the page counter for the next request
      i++;
    }

    return allCommits;
  } catch (error) {
    console.error("Error fetching commits:", error.message);
    throw error; // Rethrow the error for the caller to handle
  }
};

/**
 * Search for a Github repository using a partial search over the repository name passing input as search criteria
 *
 * @param {Object[]} commits - List of commits to be grouped.
 * @returns {Object[]} - List of objects representing weekly commit data.
 */
export const searchRepositories = async (input, demo = false) => {
  const threeMonthsAgo = new Date();
  threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);
  const formattedDate = threeMonthsAgo.toISOString();

  try {
    if (demo) {
      console.log(`demo: ${searchRepoDemo}`);
      const response = searchRepoDemo;
      const repositories = response.data.items;
      return repositories;
    } else {
      const response = await octokit.request("GET /search/repositories", {
        headers: {
          "X-GitHub-Api-Version": "2022-11-28",
        },
        q: `${input} in:name`, // Search for repositories with INPUT in the name
        sort: "stars", //I selected this as my own criteria since we want top repositories as first suggestions.
        order: "desc", // Specify the order (descending)
        per_page: 6, // Number of results per page
        page: 1, // Page number
        since: formattedDate, // reduce execution timne
      });

      // Extract and log the repository data
      const repositories = response.data.items;
      return repositories;
    }
  } catch (error) {
    console.error("Error:", error.message);
  }
};

// Fetch the last issue of the repository
export const fetchLastIssue = async (owner, repo, demo = 0) => {
  let title = "";
  let body = "";
  let response = {};
  try {
    if (demo>0) {
      response = issueDemo[demo-1];

    } else {
      const response = await octokit.request(
        "GET /repos/{owner}/{repo}/issues",
        {
          owner: owner,
          repo: repo,
          sort: "created",
          direction: "desc",
          headers: {
            "X-GitHub-Api-Version": "2022-11-28",
          },
        }
      );
    }

    // Check if there's at least one issue
    if (response.data) {
      title = response.data[0].title;
      body = response.data[0].body;
      return title + " | " + body;
    }
  } catch (error) {
    console.error("Error fetching last issue:", error);
    // Handle errors as needed
  }
};

// Fetch the last issue of the repository
export const fetchDescription = async (owner, repo, demo = false) => {
  let response = {};
  try {
    if (demo>0) {
      response = await descriptionDemo[demo];
    } else {
      response = await octokit.request("GET /repos/{owner}/{repo}", {
        owner: owner,
        repo: repo,
        sort: "created",
        direction: "desc",
        headers: {
          "X-GitHub-Api-Version": "2022-11-28",
        },
      });
      console.log(`----fetchDescription: ${JSON.stringify(response)}`);
    }

    let description = "";
    // Check if there's at least one issue
    if (response.data) {
      description = response.data.description;
      return description;
    }
  } catch (error) {
    console.error("Error fetching last issue:", error);
    // Handle errors as needed
  }
};

/**
 * Search for a Github repository using a partial search over the repository name passing input as search criteria
 *
 * @param {string} repo - Repository name
 * @param {string} owner - Owner of the repository
 * @returns {Object[]} - List of objects containning the forecast, sentiment analysis and project description
 */
export const getRepoData = async (owner, repo, demo = 0) => {
  try {
    // Fetch all commits for the specified repository
    let commitsData = [];
    let plotData = [];
    if (demo>0) {
        if(demo ===1){
            commitsData = pytorchCommits;
        }else if(demo ===2){
            commitsData = reactCommits;
        }else{
            commitsData = phpCommits;

        }
      // commitsData = commitsData;
      plotData = weeklyCommits(commitsData);
    } else {
      commitsData = await getAllCommits(owner, repo);
      plotData = weeklyCommits(commitsData);
    }

    let formatBackendPost = {
      dates: plotData.map((data) => data.startDate),
      commits: plotData.map((data) => data.totalCommits),
    };
    let forecastSubprocess = false;
    let formatedData = getPlotFormat(formatBackendPost, forecastSubprocess);

    return formatedData;
  } catch (error) {
    console.error("Error fetching commits:", error.message);
    // Handle errors (e.g., display an error message to the user)
  }
};

/**
 * Search for a Github repository using a partial search over the repository name passing input as search criteria
 *
 * @param {string} repo - Repository name
 * @param {string} owner - Owner of the repository
 * @returns {Object[]} - List of objects containning the forecast, sentiment analysis and project description
 */

export const getRepoForecast = async (series, demo = 0) => {
  try {
    let forecasts = await getForecast(series);
    let forecastSubprocess = true;
    let forecastsFormated = []
    for(let i =0; i < forecasts.length; i++){
        let forecast = forecasts[i];
        let formattedForecast = getPlotFormat(forecast, forecastSubprocess);
        forecastsFormated.push({ ...formattedForecast, id: i});
    }
    return forecastsFormated;
  } catch (error) {
    console.error("Error fetching commits:", error.message);
    // Handle errors (e.g., display an error message to the user)
  }
};

/**
 * Search for a Github repository information and pass it as input for the llm models
 *
 * @param {string} repo - Repository name
 * @param {string} owner - Owner of the repository
 * @returns {Object[]} - List of objects containning the forecast, sentiment analysis and project description
 */
export const getRepoInfo = async (owner, repo, demo = 0) => {
  try {
    // Fetch LLM Infor
    let repoIssue = "";
    let repoDescription = "";
    let sentimentCategories = ""
    repoDescription = await fetchDescription(owner, repo, demo);
    let llmDescription = await getLLMDescription(repoDescription);

    repoIssue = await fetchLastIssue(owner, repo, demo);
    sentimentCategories = await getLLMSentiment(repoIssue);

    let response = {
      llmDescription: llmDescription,
      sentimentCategories: sentimentCategories,
    };

    return response;
  } catch (error) {
    console.error("Error fetching commits:", error.message);
    // Handle errors (e.g., display an error message to the user)
  }
};

/**
 * Gets user information by sending a request to the server.
 *
 * @param {string} user_id - The user ID.
 * @returns {Promise<Object>} - A promise that resolves to the user information.
 */
export const getForecast = async (commitsData) => {
 
  let dataSeries= []
  try {
    for (let id = 0; id < commitsData.length; id++) {
        let dates = commitsData[id].labels
        let commits = commitsData[id].datasets[0].data;
        let data =  {
            dates: dates,
            commits: commits,}
            dataSeries.push(data)
    }
    
    const response = await fetch(
      `http://localhost:8000/repositories/get_forecast`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            dataSeries: dataSeries,
        }
        ),
      }
    );
    var data = await response.json();
    return data;
  } catch (error) {
    return {
      response: "**Sorry for the inconvenience - user** \n" + error,
    };
  }
};

/**
 * Takes the repository description and send it to the  backend Langchain ChatGPT application
 *
 * @param {string} repoDescription - RepositoryDescription
 * @returns {Promise<Object>} - A promise that resolves to the user information.
 */
export const getLLMDescription = async (repoDescription) => {
  try {
    let formatedQuery = "*** " + repoDescription;
    const response = await fetch(
      `http://localhost:8000/langchain/nlp_description`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: formatedQuery,
        }),
      }
    );
    var llmDescription = await response.json();
    return llmDescription.description;
  } catch (error) {
    return {
      response: "**Sorry for the inconvenience - user** \n" + error,
    };
  }
};

/**
 * Gets repository last Issue and returns the sentiment analysis of the message.
 *
 * @param {string} repoIssue - Last issue of the repository obtained with GithubAPI
 * @returns {Promise<Object>} - A promise that resolves to the user information.
 */
export const getLLMSentiment = async (repoIssue) => {
  try {
    const response = await fetch(
      `http://localhost:8000/langchain/issue_sentiment`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: repoIssue,
        }),
      }
    );
    var sentimentCategories = await response.json();
    console.log();
    return sentimentCategories.category;
  } catch (error) {
    return {
      response: "**Sorry for the inconvenience - user** \n" + error,
    };
  }
};

export const interactLLM = async (conversationId, message) => {
  try {
    const response = await fetch(
      `${process.env.REACT_APP_API_URL}/${getConversationType()}/stream`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: conversationId,
          message: message,
        }),
      }
    );
    const reader = response.body.getReader();
    return reader;
  } catch (error) {
    return {
      response:
        "**Sorry for the inconvenience, I have encountered an error. Please try again later!**",
    };
  }
};
