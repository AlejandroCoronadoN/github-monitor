import pandas as pd
import os
import math
import random
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from datetime import datetime


def get_representative_sample(repo_names, sample_size=10, seed=42):
    """
    Get a representative sample of repositories from the provided list.

    Parameters:
    - repo_names (pd.Series): Pandas Series containing repository names.
    - sample_size (int): The size of the representative sample.
    - seed (int): Seed for the random number generator.

    Returns:
    - list: A list containing a representative sample of repository names.
    """
    # Set the seed for reproducibility
    random.seed(seed)

    # Check if the sample size is greater than the total number of repositories
    if sample_size > len(repo_names):
        raise ValueError("Sample size cannot be greater than the total number of repositories.")

    # Get a representative sample using random sampling
    sample = random.sample(repo_names.tolist(), sample_size)

    return sample


import math

def calculate_sample_size(population_size: int, confidence_level: float = 0.95, margin_of_error: float = 0.01) -> int:
    """
    Calculate the required sample size for a given population. Assumes a normal distribution and exclude values that fall outside 95% of the distribution.

    Parameters:
    - population_size (int): The size of the population.
    - confidence_level (float, optional): The desired confidence level, default is 0.95.
    - margin_of_error (float, optional): The acceptable margin of error, default is 0.01.

    Returns:
    - int: The calculated sample size.
    """
    # Z-score for a given confidence level (default is for 95% confidence)
    z_score = 1.96

    # Assuming a conservative estimate for the proportion (p)
    p = 0.5

    # Calculate the numerator and denominator for the sample size formula
    numerator = z_score ** 2 * p * (1 - p)
    denominator = margin_of_error ** 2

    # Calculate the sample size using the formula
    sample_size = math.ceil((numerator / denominator) / (1 + ((numerator - 1) / population_size)))

    return sample_size



def format_date(df):
    df['week_number'] =  df['week_number'].apply(lambda x: int(x))
    df['year'] =  df['year'].apply(lambda x: int(x))
    df['date'] = pd.to_datetime(df['year'].astype(str) +df['week_number'].astype(str) + '1', format='%Y%W%w')
    # Find the closest Sunday for each date
    df['date'] = df['date'] + pd.to_timedelta((6 - df['date'].dt.dayofweek) % 7, unit='D')
    return df

def preprocess_data(df_raw):
    df_regroup = df_raw.groupby(["repo_author_single", "year", "week_number"])["commit_count"].sum()
    df_regroup = df_regroup.reset_index()
    unique_repo_names =df_regroup.repo_author_single.unique()

    total_repositories = len(df_regroup)
    sample_size = calculate_sample_size(total_repositories)
    logging.info(f"Recommended sample size: {sample_size}")

    # Example usage
    # Assuming you already have a Pandas Series named unique_repo_names
    subset_sample = get_representative_sample(unique_repo_names, sample_size=sample_size, seed=123)

    # Print the representative sample
    logging.info(len(subset_sample))
    subset_sample[:10]

    df_regroup = format_date(df_regroup)
    #Make sure that we are using Sunday as the first day of the week, this is the default week start for the shift and window functions

    #Filter dates.
    date_string = '2023-01-01'
    max_date = datetime.strptime(date_string, '%Y-%m-%d')
    five_years_ago = max_date - pd.DateOffset(years=6)
    df_regroup = df_regroup[(df_regroup.date >= five_years_ago)&(df_regroup.date <= max_date)]


    sorted_dates = sorted(df_regroup.date.unique())
    if sorted_dates[0].dayofweek != 6:
        raise ValueError("Weeks are not starting on Sunday")

    # Print the representative sample
    logging.info(len(subset_sample))

    df_sample = df_regroup[df_regroup.repo_author_single.isin(subset_sample)]
    df_sample.columns
    mean = df_sample['commit_count'].mean()
    std = df_sample['commit_count'].std()
    outlier_benchmark = mean + (2*std)
    max_commits_week =df_sample['commit_count'].max()

    logging.info("\n-Outliers:\noutlier_benchmark:" , outlier_benchmark)
    logging.info("max_commits_week:" , max_commits_week)
    logging.info("mean:" , mean)

    #Apply outlier imputation
    df_bounded = df_sample.copy()
    df_bounded.loc[df_bounded["commit_count"]>=outlier_benchmark, "commit_count"] = int(round(outlier_benchmark))

    df_group_repo_bounded = df_bounded.groupby(["repo_author_single"])["commit_count"].sum().reset_index()
    df_test = df_group_repo_bounded[df_group_repo_bounded.commit_count<=100]

    df_bounded = df_bounded.rename(columns= {"repo_author_single":"repo_name"})
    len(df_bounded.repo_name.unique())
    return df_bounded


if __name__ == "__main__":

    # Get the current working directory
    current_dir = os.getcwd()

    # Navigate to the data directory from the forecast directory
    data_path = os.path.join(current_dir, "../data/raw/commit_history_raw.csv")

    # Read the CSV file
    df_raw = pd.read_csv(data_path)
    df_bounded =preprocess_data(df_raw)

    output_path= os.path.join(current_dir, "../data/preprocess/commit_history_subset.csv")
    df_bounded.to_csv(output_path, index=False)

    testing_sample = df_bounded.repo_name.unique()[:150]
    df_testing = df_bounded[df_bounded.repo_name.isin(testing_sample)]
    logging.info("Max commits per day: ", df_testing.commit_count.max())
    logging.info("Max date: ", df_testing.date.max())
    logging.info("Min date: ", df_testing.date.min())
    logging.info("Shape: ", df_testing.shape)

    testing_path= os.path.join(current_dir, "../data/preprocess/commit_history_subset_test.csv")
    df_testing.to_csv(testing_path)
