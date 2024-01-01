# Project Planning and Decision Document

## Machine Learning - Full Stack Development

### Feature:
- **Forecasting GitHub Commit Activity using TensorFlow.js and Python Backend**

### Justification:
- I chose this ML-powered feature because it adds predictive analytics to the project. It enables users to anticipate future trends in GitHub commit activities for a more informed decision-making process.
- For the technology stack, I will develop a FastAPI - React application that will allow me to divide my project into frontend and backend. By starting with the frontend, I will have a grasp of all the potential blocks, the input format, and small details that will be required when connecting my application with the real backend and yet have a functional version of the project even before finishing the entire Python backend.
    - **TensorFlow.js:** Being a versatile library, it allows for the efficient implementation of machine learning models directly in the browser, enhancing the user experience. This JS integration will reduce the executing time for forecasting and linear regression analysis by training simple models and making predictions. I will use the output of this library to create a fast POC.
    - **Python Backend:** To make a further analysis of GitHub Public repository commit history, I will use pandas and scikit-learn to create better features that will improve forecast performance. Furthermore, I will test different ML models to provide insights about the time series behavior and provide an ensemble model with the final predictions for our forecasting exercise.
    - **Chart.js:** I will use this library for the frontend plotting component, as it is one of the libraries with more support for small datasets plotting.
    - **Vite:** For the frontend project structure, I will use Vite to take advantage of its speed and simplicity, providing a smooth development experience.

- Finally, I'll be focusing on ML Time series forecast as it is my area of expertise, and I'm aware of the potential blocks for TimeSeries regressions.

---

## Project Risks Assessment

### Greatest Areas of Risk:

1. **Data Availability and Quality:**
   - Risk: Unreliable or insufficient data from the GitHub API could impact the accuracy of the commit history analysis and the subsequent forecast.
   - Mitigation: I will look for available GitHub Datasets that allow me to train my models and isolate the model training and tuning process from the production side of the application. I will use my trained model in my FastAPI endpoint; that way, I'm able to test and use different models without breaking the application functionality.

2. **Model Performance and Generalization:**
   - Risk: The TensorFlow.js and scikit-learn model may not generalize well to different repositories or may not capture the complexity of commit patterns.
   - Mitigation: First, I will start with regression and random forest models that have the characteristic of being trained really fast and can also make fast predictions over the testing data set. As I advance into the problem, I'll be adding more models to improve the accuracy of my forecast. Second, I will be focusing on autoregressive features for the feature engineering process. In particular, I'll be using window functions that allow us to get the average (exponential and weighted) of the N previous months with an offset of M months. Example: feature_avg3months_lag6months will give us a series of points with the average. By using only variables that depend on the commit history, I will avoid dealing with ad-hoc information from each repository that might not be available for all users. Moreover, the design of these utility tools will become useful to create additional variables in the future and improve the model performance. Also, I will create clusters using TimeSeries distribution that, hopefully, will encompass repository characteristics that will not be passed directly as part of the model input (For example, I would expect my clusters to group language or application types in different clusters).

3. **Solving the main goal of the assignment**
   - Risk: We need to remember that the objective is to provide useful information to the user to discriminate between different repos and select the one that will be more useful and will have more support in the future. Commit predictions are only a proxy for this variable (long-term-support-variable). Even if the forecasting is accurate enough, it will only tell us how the repository will behave in the future and not how well it will be adopted by the community.
   - Mitigation: One way to solve this problem is to create a longer forecast that covers more than 3 months; this way, the probability of a repository being successful after a 6-month (or 1 year) positive commit trend forecast will be bigger than a 3 months one. Moreover, additional information can be incorporated into the model to address this problem, like characteristics about the repository or GitHub user or even design a new model that compares two repositories' histories and provides the answers directly to the user's need (For example, provide two repositories commit history and answer which one will have more commits in the following L months).

4. **Forecasting model obstacles:**
   - Risk: Forecasting requires at least 1 year of historic information to detect variables yearly and monthly trends. Lifespan of the repositories might change the outcome drastically and lead to wrong conclusions. Take for example a new library for GoLang that when it was launched it had thousand of commits and at some point, it seemed to be the obvious choice for a long-term support library. However, this trend drastically went down after a couple of years of GoLang being in the market.
   - Mitigation: Provide at least 1 year of data to train the models and create control variables for month, year, and repository age. Include a final decision criteria with the help of AI tools like LangChain and GPT that can use both, the commit forecast of two different repos and additional information available in Google to provide reasons to choose one repository over another.

5. **User Adoption and Engagement:**
   - Risk: Users may not find value in the ML-powered feature, leading to low adoption rates and engagement. The repository decision process is not homogeneous for all users. Some might be more concerned about using something similar to what they have used before, or there might be some tools and API available in the market that makes this process much simple.
   - Mitigation: Create additional features that make a good differentiator from all the other tools in the market. After small research, I couldn't find any tool that interacts with the user in a more personal way. In other words, creating more user-friendly applications could lead to more engagement.

---

## Design Considerations

### Changes/Additions to Design:

1. **User-Friendly Interaction:**
   - **Problem:** At first hand, it's difficult to understand what the application is doing and what it is useful for.
   - **Solution:** Creating a dynamic tutorial inside the tool could help and guide the user in comparing two different repositories, go through all the features of the tool, and eventually letting her know which repository is the best option for her use case. All of this can be really difficult to infer from a simple plot.

2. **Discrimination Criteria:**
   - Add additional information to the application to understand what the user is looking for. For example, if I want to look for the best option for a python application, I might want to constrain the type of repositories that I'm able to search in the search bar. As I already mentioned, we could also provide contextual information directly from google or wikipedia, that helps the user understand what each library does and how it is supposed to be used.

3. **Interactive Elements:**
   - Implement interactive elements to allow users to explore specific time frames, repositories, or customize the parameters of the forecast.

---

## Future Feature Considerations

### Potential Future Features:

1. **User Authentication and Personalization:**
   - Implement user accounts to save preferences, track historical analyses, and provide personalized recommendations based on the user's interests.
   - Use user's information to provide more information to the user taking into account the projects and programming languages that he has used in the past.

2. **Integration with GPT and LangChain:**
   - Using context to answer user question could be a differentiator and also will address the problem from more than one angle, giving the user more information to choose the best repository for his needs.

3. **Personalized Github milestones:**
   - With the same architecture, data processing, and frontend we can easily create an additional feature to create user-level forecast. For example, how many commit will a particular user make over the following L months. The model will be retrained at a user-repository (much more information to process). After this you can add your forecast and your suggested forecast for example how many commits will make Rodrigo Medina in the following 6 weeks in a new React repository and what would be your own personal forecast. If you make more than him, he will invite the pizza, at least that's what I heard.

4. **BONUS | Give me my Github horoscope:**
   - Recently I been playing creating stories and content with LLM models. I really like when you combine real information with fictional data allowing the LLM models to make something more creative. I would like provide the LLM model my commit forecast for this project and my next 3 months horoscopes so it can tell me an interesting story of why I will make a lot of commits over the following 8 hours and then I will stop for the following month, and then again a lot more by the end of January. Great opportunities are coming, or so they say.

---

## Clarifying Questions

1. **Data Sensitivity:**
   - Does the project involve handling sensitive information, and if so, what measures are in place to ensure data privacy and security?
   - Assumption: We are not concerned about passing personal information to feed the application or the ML models. I will try to avoid as much as possible using personal information.

2. **Backend and ML processing:**
   - Can python be used for the BackEnd?
   - Yes, still, I'll develop a TensorFlow.js model to showcase the use of ML with only JS.

3. **Forecasting Processing power:**
   - Model processing power is not a concern for the development of this project. Accuracy will be prioritized over processing time.
   - Even with a pretrained model, making forecast predictions still requires processing power. This is particularly true for iterative predictions over a long forecast window.
   - This problem can be easily solved in production by deploying scaling the application both vertically and horizontally.
   - We are not expected to get more than 100 queries per minute.
   - For demonstration purposes, I will address the concerns about processing time for the ML models, but I will not address any issue regarding Availability and performance at a production level.

---

## Assumptions

1. **API Availability:**
   - It is assumed that the GitHub Repositories and Repository Statistics APIs will be consistently available for fetching commit data.
   - In particular, this application doesn't expect to get more than 10,000 requests every minute.

2. **Browser Compatibility:**
   - The application is designed to be compatible with modern web browsers, and users are expected to use up-to-date browsers for optimal performance and security.

3. **Testing framework is not mandatory:**
   - I will cover edge cases for my development process, but I will not provide any testing environment like Jezt to keep maintenance of the application.
