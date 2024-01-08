### github-monitor
This is the github-monitor application created for the metalab coding challenge.

### Getting started
To start the project you will need to run the following commands.
For the backend start by creating a new environemnt for the project. I'm using conda with pyhton=3.10 but you can also use pyenv. After installing pyenv or conda install poetry to create the environemnt for the project using the pyproject.toml file. Starting from the root of the project run the following commands.
```
conda create -n github-monitor python=3.10
conda activate github-monitor
```

After installing conda we can start by installing poetry in the new environment. Poetry will handle all the libraries dependencies and install everything we need to run the project from the backend. To use poestry first we need to move to be a the root of the directory where the pyproject.toml file is located and run the following commands

```
conda install -c conda-forge poetry
poetry install
```
After all the libraries are installed we can go back and setup the of the frontend directory at ./frontend. Here you will see the package.json file which contains all the libraries that we need to run the project. Use npm to install them and the build the project to be used by the backend
```
npm install
npm run build
```

Go back to the root directory and navigate to the source folder of the fastAPI ./src application. Now that all the python dependencies are installed we can start the project and test the application with the following command.
```
uvicorn githubmonitor.main:app --reload --host 0.0.0.0
```

Go to the application at localhost http://localhost:8000/




# Architecture
The following diagram represents the application workflow. On the left side we can identify and EC2 instance dedicated to model tunning and model tarinning. This instance can be scalled vertically and include as many cores to satisfy the model tunnig and trainning workload. On the right side we have a TEST instance that is dessigned only to serve user request and deploy the application into our URL. The user is restricted to interact with the public subnet alone, that way he will never interact with the ML backend. For the test instance, we rehuse pretrained models from the DEV instance. Both instances have their own DataBase that includes three tables:
* **Users**: Designed to store user_id, name and personal information.
* **Sessions**: Table that connects users with a unique session_id. This way the user can start a new search everytime he loggins.
* **Repositories**: Stores user searches, github metadata and forecats predictions. If we store each session we can retrieve data from this table instead of calling the /get_forecast everytime.
* NOTES: These databases are not currently connected to the application. This diagram represents the production version of the app.

![Image Alt Text](./images/Forecast%20Architecture.png)


# Frontend
# Backend

## Backend - FastAPI
I'm connecting the frontend and the backend using FastAPI. I load the React compiled version to the home endpoint defined at github-monitor/src/githubmonitor. FastAPI also connects the ensemble model predictive function to an endpoint. When the user interacts with the front end by searching a new repository and the selecting it (adding it to the Repository Selection component) a new POST request is send to the /get_forecast endopoint. This endpoint make predictions with the stored pretrainned models and returns the answer to the front end.

## Backend - DataBase System
The DataBase system is designed to store user sessions. When the user interacts with the application and searches the first repository the user_id (hardcoded to 10001 by default and for each interaction) the system identifies if the user was logged before. If not, a new entry is created in the users Table. Moreover, a new entry is created in the session Table. Everytime, a session is created the repositories selected by the user are stored in the repositories session. if the user deletes a repository and search it again, the information will be available at the Repositories Table and can be loaded without needing to reprocess it.


## Backend - Forecast
This project contains a forecasting process

The back end is devided in two parts. The first section of the backend is designated for Machine learning model trainning and model evaluation. This code will live at /src, and it will be divided in three sections (folders).
* **data**: Contains the code to download data from BigQuery and obtain time series variables. These variables will allow us to create new variables and train the models.
*  **visualization**: In this folder we wil find two notebooks.
*  * **exploratory_anlaysis**: This notebook provides a couple of plots that allows us to study the behaviour of commits over time. We will use this information to create clusters that will become useful on the last step of the model evaluation and also we will make a simple exploratory anlysis of repositories description.
*  * **model_evaluation**: After training the models and cretaing predictions over the testing set, we want to know how accurate our preductions where and compare our weekly predicted commits to the actual commits associated with each repository.
*  **features**: Composed by a single script, features contain a set of functions that allow us to create autorregressive variables and windows mean features that are usefull when evaluating and making predictions over timeseries models. These functions will be use further when creating iterative predictions for the final forecast.
*  **models**: We are using two layers for the training process of the forecast.
*  * **hyperparamter_optimization**: In this script we train three models, xgboost, lightgbm and random forest. For each model we evaluate a set of hyperparameter to create a simple model tunning using a Grid search. Grid search trains the models with different combinations of hyperparameters for each model and selects the one with the best results over the evaluation set in a cross-validation algorithm.
*  * **iterative_prediction**: Once the "best" parameter for the ML models have been found, we can use these pretrained models top create actual predictions in a three month window forecast. I will use previous predictions to make a new prediction. I will repeat these process 12 times to obtain the prediction over the following three months (12 weeks)
*  * **forecast_ensemble**: Finally, the last stage of the predictive algorithm, will be an ensemble model. The objective of this last stage is to reduce the number of variables to create a forescast, only using the predictions of the lightgbm, xgboost and randomforest models and other categorical variables. The elastic net model works as a ridge and lasso regresison combined, so any irrelavant variables for the regression analysis will be deleted and we can generalize better by creating a simple linear combination of all the available models.

### Data Extraction
For the regression analysis we are going to use a subsample of the total repositories currently published in GoogleCloud, BigQuery. From a random sample of 9,900 repositories, I'll obtain the commit history of every repository and cerate a new DataSet that reports commits on a weekly level. We are going to focus on timeseries data  alone to simplify the fetaure engineering, model evaluation and model prediction process. For a more detailed description of this stage, please read the extarction.ipynb notebook at src/data.

https://www.kaggle.com/code/coronate/github-monitor-data-extraction

### Feature Enginnering
The logic of this script is very simple once the script is decompossed into single steps.
* First I proviode two lists
*  * lag_list: This list contains a set of inetgers, each integer will be passed as an argument to the interact_categorical_numerical function. This first argument represents all the lags that will be used in the model to make predictions.
*  * rolling_list: This argument determines the bounds where the interact_categorical_numerical will operate. This means tha if the window is of size 5 then we will take 5 weeks to create an average.
*  * Example: For lag_list = [0, 2, 4, 6, 8] rolling_list = [1, 2, 4, 6]

* **interact_categroical_numerical**: I already cover the porpuse of this function. Yet, is important to mention that we will be executing this function twice:
*  * **Rolling mean**: This is a simple average where the commits of each weak has the same weight in the calculated average.
*  * **EWM Mean**: Exponential mean assigns a different weight to each week, giving more importance to the weeks that fall closser to the present. This functions has shown better results and bigger correlation to the objective variable.
*  * **lag_list and rolling_list**:I already talk about this function but it's important to highlight that the number of variables returned by this process ill be a combination of the existing elements in both lists. the number of varibles increase exponentially with the number of elements in each list. This is why, I'll be only using a set of integer that represent important breakpoint in the context of the proble. rolling_list = [0, 1, 2, 4, 6] represents only the lag variable (0), the mean of commoits during the last week (1), average mean of the last two weeks, average of the last month (4) and so on. The selection of these parameters is just an heuristic that makes sense. I the future we can create larger lists and then create a ridge regression to find the best variables for this problem.

### Hyperparameter Optimization
* **KFold for time series**: In this script we use a K-fold validation for time series. This K-fold version works the same way as a normal K-fold algirthm but insetad of taking random variables from the tarining set, each iteration works with sequantial data that leaves out the last T obsetvations. For example, if we are making predictions over the follwing 12 weeks then T will be equal to those 12 last weeks. The first iteration we will predict the first wee and compare the results with the first wee contained in T. Then we repeat again trainning the model and leaving this time only T-1 weeks out of the trainning set. After repeating the process 12 times we will be able to pass all the weeks to the models and evaluated 12 times with evaluation sets.
* **Grid Search**: The K-fold evaluation will be repeated X times, where X is the number of combinations of hyperparamters that we want to test. This follows a similar logic than the feature_engineering step but this time we repeat the tarinning step for each element in every list.
*  **Random Forest**: For exmaple, the tested hyperparameters for the random forest are n_estimators, max_depth, min_samples_split, min_samples_leaf and max_features.
```     "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt"],
```
* **Saving Mdoels**: After trainning the models and finding the best set of hyperparameters, we would like to save that model as a joblib object. That way we can call the pretrained model and use it for the last prediction of our forecats.
* **Model Evaluation**: Along with the trainning procedure, we are saving important variables that will give us insights about the predictive power of the different ML models. We want to identify the cases and scenarios where a particular model is stronger and user them accordingly.

### Iterative Prediction
After storing the best model in the models folder, we can rehuse them to make predictions sequantially. So far, we have taken advantage of the available information to contrast ourt prediction with the actual data. However, with real data we are not expected to know what is going to happen over the following week and what happens after that week. For this problem, we create iterative predictions that allows us to keep feeding the model and create new predictions on a more distant future by using our previous predictions.
