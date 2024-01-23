# Metalab github-monitor coding challenge
This is the github-monitor application created for the metalab coding challenge.


## Getting started - Development

To start the project you will need to run the following commands.
For the backend start by creating a new environemnt for the project. I'm using conda with pyhton=3.10 but you can also use pyenv. After installing pyenv or conda install poetry to create the environemnt for the project using the pyproject.toml file. Starting from the root of the project run the following commands.
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

conda create -n githubmonitor python=3.8.10
conda activate githubmonitor
```


### Python packages
After installing conda we can start by installing poetry in the new environment. Poetry will handle all the libraries dependencies and install everything we need to run the project from the backend. To use poestry first we need to move to be a the root of the directory where the pyproject.toml file is located and run the following commands. It's very important to use python 3.11 to run all the dependencies for this project.
```
conda install conda-forge::poetry
poetry install
sudo apt install uvicorn

```


After all the libraries are installed we can go back and setup the of the frontend directory at ./frontend. Here you will see the package.json file which contains all the libraries that we need to run the project. Use npm to install them and the build the project to be used by the backend.

Install nvm to use the correct version of npm for this project (npm 18.19)
```
sudo apt-get install curl
curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash
source ~/.bashrc

nvm install node
nvm install 18
```
Move to the github-repository/frontend folder and execute the following commands to install all node modules and create a compiled version of the frontend.
```
npm install
npm run build
```

Now the application is ready to be tested but you will need to add your own github and ChatGPT credentials to start using all the features of this project. Firt add you own Github credentials by replacing this line inside the .env file
```
VITE_GITHUB_API_TOKEN='ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
REACT_APP_API_URL = 'http://localhost:8000'
```
Now modify the configuration.py file and replace the chatgpt_api_credentials by replacing this section of the code
```

    class Config:  # noqa: D106
        import os

        env_file = os.path.join(os.path.dirname(__file__), ".env")

    app_name: str = "Github Monitor"
    log_level: str = "info"
    openai_api_key: str = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" #<-This line
    max_message_history: int = 5
    temperature: float = 0.4
    agent_model_name: str = "chatgpt-3.5-turbo"
    version: str = "0.0.1"
    minimum_delay_ms: int = 5
    maximum_delay_ms: int = 10

```

Go back to the root directory and navigate to the source folder of the fastAPI ./src application. Now that all the python dependencies are installed we can start the project and test the application with the following command. The application will start at port 800 and you can use all the features now.
```
uvicorn githubmonitor.main:app --reload --host 0.0.0.0
```

Go to the application at localhost http://localhost:8000/ and start searching new repositories.

Note: If the project is being runned directly from the repository, you will need to create all the models and go trough the Machine Learning trainning pipeline to enable the forecast features and allow the application to work. The main.ipynb Jupyter notebook to complete this ML trainning process.



# Architecture
The following diagram represents the application workflow. On the left side we can identify and EC2 instance dedicated to model tunning and model tarinning. This instance can be scalled vertically and include as many cores to satisfy the model tunnig and trainning workload. On the right side we have a TEST instance that is dessigned only to serve user request and deploy the application into our URL. The user is restricted to interact with the public subnet alone, that way he will never interact with the ML backend. For the test instance, we rehuse pretrained models from the DEV instance. Both instances have their own DataBase that includes three tables:
* **Users**: Designed to store user_id, name and personal information.
* **Sessions**: Table that connects users with a unique session_id. This way the user can start a new search everytime he loggins.
* **Repositories**: Stores user searches, github metadata and forecats predictions. If we store each session we can retrieve data from this table instead of calling the /get_forecast everytime.
* NOTES: These databases are not currently connected to the application. This diagram represents the production version of the app.

![Image Alt Text](./images/Forecast%20Architecture.png)

# Demo
The following media shoiwcase the three different ML features of this project and the Reactt UI:
* **React ChartJS: **: Designed to store user_id, name and personal information.
* **Forecast: **: Ensemble model for RandomForest, Xgboost and elasticnet. TimeSeries cluster predictive enhancement.
* **Sentiment Analysis **: Reviews the last GitHub issue and returns a sentiment analysis of the issue. The issue is classified and depending on the assigned category, a representative emoji is provided.
* **NLP Description **: Use the LangChain LLM framework to create bfrief descriptions for the github repositories.

![Image Alt Text](./images/demo.gif)

# Development process
For the development process I created two branches in the github repository. First I solved most of the frontend logic and a basic rendering that allowed me to test the backend and slowly integrating the application. For the backend I cretaed a new branch and started working with my BigQuery information. All the model tunning process was made in individual scripts that allowed me to solve individual tasks and then validate my information before going to the next step. After reviewing and evalauting my results I was able to cretae a production version of the ML pipeline that used the models generated in the tunning process. The FastAPI get_forecast endpoint summarizes all the ML process by rehusing the functions of the feature_engineering, iterative_prediction and forecast_ensemble scripts. I was able to test all the endpoint trough the FastAPI docs extension (http://localhost:8000/docs) before integrating the application. On an intermediate step I created a new branch and connected the front end and backend. In this stage of the development process I had to connect the frontend functions (located at utils.js) to the correct FastAPI endpoint. I started with the get_forecast function and proceed with the langchain route endpooints. Finally, I worked refactoring my code by providing more documentation and adding more dtails to the frot end to replicate the expected behaviour.

![Image Alt Text](./images/gitprocess.png)


# Backend

## Backend - FastAPI
I'm connecting the frontend and the backend using FastAPI. I load the React compiled version to the home endpoint defined at github-monitor/src/githubmonitor. FastAPI also connects the ensemble model predictive function to an endpoint. When the user interacts with the front end by searching a new repository and the selecting it (adding it to the Repository Selection component) a new POST request is send to the /get_forecast endopoint. This endpoint make predictions with the stored pretrainned models and returns the answer to the front end.




## Backend - Forecast

The project includes a forecasting process, with the backend divided into two main sections: Machine Learning Model Training and Model Evaluation. The backend code is organized under the directory `/src/githubmonitor/forecast` and is further categorized into three sections (folders).

* **data:**
  Contains code to download data from BigQuery and obtain time series variables, essential for creating new variables and training models.

* **visualization:**
  In this folder, two notebooks are present:
  * **exploratory_analysis:**
    Provides plots to study the behavior of commits over time. Information from this analysis is used to create clusters for the final step of model evaluation. Additionally, a simple exploratory analysis of repository descriptions is conducted.
  * **model_evaluation:**
    After training the models and generating predictions on the testing set, this notebook assesses prediction accuracy and compares weekly predicted commits to actual commits associated with each repository.

* **features:**
  Composed of a single script, this folder contains functions to create autoregressive variables and window mean features. These features are useful for evaluating and making predictions in time series models and will be further used in creating iterative predictions for the final forecast.

* **models:**
  The training process for the forecast involves two layers:
  * **hyperparameter_optimization:**
    Trains three models - xgboost, lightgbm, and random forest - with hyperparameter tuning using a Grid Search. The best-performing combination of hyperparameters for each model is selected based on cross-validation results over the evaluation set.
  * **iterative_prediction:**
    Uses the pre-trained models to create actual predictions in a three-month window forecast. The process involves using previous predictions to inform new predictions, repeated 12 times for a three-month (12-week) outlook.
  * **forecast_ensemble:**
    The final stage involves an ensemble model. This step aims to reduce the number of variables for forecasting, utilizing predictions from lightgbm, xgboost, and random forest models, along with other categorical variables. The elastic net model combines ridge and lasso regression, eliminating irrelevant variables for better generalization through a simple linear combination of available models.

For more detailed information about the backend process and model training, please refer to the `main.ipynb` file located at `arc/githubmonitor/forecast/main.ipynb`. This notebook provides a detailed description of the process and offers a user-friendly interface to interact with the code and models.

