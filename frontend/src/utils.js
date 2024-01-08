/**
 * Module: GitHub API Utilities
 * Description: This module contains functions for interacting with the GitHub API,
 * retrieving commit data, and formatting it for chart presentation.
 * @module githubApiUtils
 */

import { Octokit } from "octokit";
import commitsData from "./commitsData.json"
// GitHub API token for authentication
const githubApiToken = import.meta.env.VITE_GITHUB_API_TOKEN;

// Initialize Octokit instance with authentication
const octokit = new Octokit({
    auth: githubApiToken
});

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
}




/**
 * Formats forecast data for chart presentation.
 *
 * @param {Object[]} plotData - List of objects containing commit data.
 * @returns {Object} - Formatted data suitable for charting.
 */
export const formatForecastData = (response) => {

    const formattedDates = response.dates.map(timestamp => {
        const date = new Date(timestamp / 1e6); // Convert nanoseconds to milliseconds
        return date.toLocaleDateString('en-US'); // Adjust the locale as needed
    })


    let formatChatData = {
        labels: formattedDates,
        datasets: [
            {
                label: "Commit History",
                data: response.forecast,
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
}


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
    let n_weeks = 80
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
            id: n_weeks-week,
            startDate: startDate.toISOString(),
            endDate: endDate.toISOString(),
            commits: weeklyCommits,
            totalCommits: weeklyCommits.length
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
            const commitsResponse = await octokit.request('GET /repos/{owner}/{repo}/commits', {
                owner:owner,
                repo:repo,
                since: formattedDate,
                per_page: 100, // Adjust per_page as needed
                page: i,
                headers: {
                  'Accept': 'application/vnd.github.v3+json', // Use the recommended Accept header
                },
              });

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
        console.error('Error fetching commits:', error.message);
        throw error; // Rethrow the error for the caller to handle
    }
};


/**
 * Search for a Github repository using a partial search over the repository name passing input as search criteria
 *
 * @param {Object[]} commits - List of commits to be grouped.
 * @returns {Object[]} - List of objects representing weekly commit data.
 */
export const searchRepositories = async (input) => {

    const threeMonthsAgo = new Date();
    threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);
    const formattedDate = threeMonthsAgo.toISOString();
    try {
      const response = await octokit.request('GET /search/repositories', {
        headers: {
          'X-GitHub-Api-Version': '2022-11-28',
        },
        q: `${input} in:name`, // Search for repositories with INPUT in the name
        sort: 'stars', //I selected this as my own criteria since we want top repositories as first suggestions.
        order: 'desc', // Specify the order (descending)
        per_page: 6, // Number of results per page
        page: 1, // Page number
        since:formattedDate// reduce execution timne
      });

      // Extract and log the repository data
      const repositories = response.data.items;
      return repositories
    } catch (error) {
      console.error('Error:', error.message);
    }
  };

    // Fetch the last issue of the repository
export const fetchLastIssue = async (owner, repo) => {
    let title = "";
    let body = "";
    try {
        const response = await octokit.request('GET /repos/{owner}/{repo}/issues', {
            owner: owner,
            repo: repo,
            sort:"created",
            direction: "desc",
            headers: {
            'X-GitHub-Api-Version': '2022-11-28'
            }
        })

        // Check if there's at least one issue
        if (response.data.length > 0) {
            title = response.data[0].title;
            body =response.data[0].body;
            return (title + " | " + body);
        }
    } catch (error) {
    console.error('Error fetching last issue:', error);
    // Handle errors as needed
        }
    };



    // Fetch the last issue of the repository
export const fetchDescription = async (owner, repo) => {
    try {
    const response = await octokit.request('GET /repos/{owner}/{repo}', {
        owner: owner,
        repo: repo,
        sort:"created",
        direction: "desc",
        headers: {
          'X-GitHub-Api-Version': '2022-11-28'
        }
      })
    let description = ""
    // Check if there's at least one issue
    if (response.length > 0) {
        description =response.data.description;
        return description
    }
    } catch (error) {
    console.error('Error fetching last issue:', error);
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
export const getRepoInfo = async (owner, repo) => {
    try {
      // Fetch all commits for the specified repository
      let test =true
      let commitsData = []
      let plotData = []
      let repoIssue = ""
      let repoDescription = ""
      if(test){
        commitsData = commitsData;
        plotData = weeklyCommits(commitsData);

      }else{
        console.log(`owner: ${owner}`);
        console.log(`repo: ${repo}`);


        plotData = weeklyCommits(commitsData);
      }
      // Set the commits data to the state
      // Group commits into weekly intervals and calculate total commits for each week
      // Format the commit data for chart presentation
      let formattedPlotingData = formatChatData(plotData);

      let formatBackendPost = {
        dates: plotData.map((data) => data.startDate),
        commits:plotData.map((data) => data.totalCommits),
        };

        repoDescription = await fetchDescription(owner, repo);
        repoIssue = await fetchLastIssue(owner, repo);
        commitsData = await getAllCommits(owner, repo);

      let llmDescription = await getLLMDescription(repoDescription)
      let sentimentCategory = await getLLMSentiment(repoIssue)
      let forecast = await getForecast(formatBackendPost)


      let formatteForecast = formatForecastData(forecast);
      response = {
        "forecast": formatteForecast,
        "llmDescription": llmDescription,
        "sentimentCategory": sentimentCategory
      }

      return response
    } catch (error) {
      console.error('Error fetching commits:', error.message);
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
    try {
      const response = await fetch(
        `http://localhost:8000/repositories/get_forecast`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dates: commitsData.dates,
            commits: commitsData.commits
        }),
        }
      );
      var data = await response.json();
      console.log(`***** data: ${JSON.stringify( data )}`);
      return data;
    } catch (error) {
      return {
        response:
          "**Sorry for the inconvenience - user** \n" + error,
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
      const response = await fetch(
        `http://localhost:8000/langchain/get_llmdescription`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: repoDescription
        }),
        }
      );
      var data = await response.json();
      console.log(`***** data: ${JSON.stringify( data )}`);
      return data;
    } catch (error) {
      return {
        response:
          "**Sorry for the inconvenience - user** \n" + error,
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
        `http://localhost:8000/langchain/get_sentiment`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: repoIssue
        }),
        }
      );
      var sentimentCategory = await response.json();
      console.log(`***** Sentiment Analysis: ${JSON.stringify( sentimentCategory )}`);
      return sentimentCategory;
    } catch (error) {
      return {
        response:
          "**Sorry for the inconvenience - user** \n" + error,
      };
    }
  };
