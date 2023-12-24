/**
 * Module: GitHub API Utilities
 * Description: This module contains functions for interacting with the GitHub API,
 * retrieving commit data, and formatting it for chart presentation.
 * @module githubApiUtils
 */

import { Octokit } from "octokit";

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
 * Groups commits into weekly intervals and calculates the total number of commits for each week.
 *
 * @param {Object[]} commits - List of commits to be grouped.
 * @returns {Object[]} - List of objects representing weekly commit data.
 */
export const weeklyCommits = (commits) => {
    // Initialize an array to store weekly commit lists
    const plotData = [];

    // Get the current date
    const currentDate = new Date();

    // Iterate for 10 weeks
    for (let week = 0; week < 52; week++) {
        // Calculate the start and end dates of the current week window
        const endDate = new Date(currentDate - week * 7 * 24 * 60 * 60 * 1000);
        const startDate = new Date(endDate - 7 * 24 * 60 * 60 * 1000);

        // Filter commits within the current week window
        const weeklyCommits = commits.filter((commit) => {
            const commitDate = new Date(commit.commit.author.date);
            return commitDate >= startDate && commitDate <= endDate;
        });

        // Add the weekly commits to the list
        plotData.push({
            id: week,
            startDate: startDate,
            endDate: endDate,
            commits: weeklyCommits,
            totalCommits: weeklyCommits.length
        });
    }

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
    const perPage = 100;
    let i = 1;
    let allCommits = [];
    let dataContinue = true;

    try {
        while (dataContinue) {
            const commitsResponse = await octokit.request('GET /repos/{owner}/{repo}/commits', {
                owner,
                repo,
                per_page: perPage,
                page: i,
                headers: {
                    'X-GitHub-Api-Version': '2022-11-28',
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
    try {
      const response = await octokit.request('GET /search/repositories', {
        headers: {
          'X-GitHub-Api-Version': '2022-11-28',
        },
        q: `${input} in:name`, // Search for repositories with INPUT in the name
        sort: 'best match', // You can change the sorting criteria if needed
        order: 'desc', // Specify the order (descending)
        per_page: 10, // Number of results per page
        page: 1, // Page number
      });

      // Extract and log the repository data
      const repositories = response.data.items;
      console.log('Search Results:', repositories);
    } catch (error) {
      console.error('Error:', error.message);
    }
  };
