"""Script version of the preprocess stage, clear outliers and impute values."""
import math
import os
import random
from typing import List

import pandas as pd


def get_representative_sample(
    repo_names: pd.Series, sample_size: int = 10, seed: int = 42
) -> List[str]:
    """Get a representative sample of repositories from the provided list.

    Parameters:
    - repo_names (pd.Series): Pandas Series containing repository names.
    - sample_size (int): The size of the representative sample.
    - seed (int): Seed for the random number generator.

    Returns:
    - List[str]: A list containing a representative sample of repository names.
    """
    # Set the seed for reproducibility
    random.seed(seed)

    # Check if the sample size is greater than the total number of repositories
    if sample_size > len(repo_names):
        raise ValueError(
            "Sample size cannot be greater than the total number of repositories."
        )

    # Get a representative sample using random sampling
    sample = random.sample(repo_names.tolist(), sample_size)

    return sample


def format_date(df: pd.DataFrame) -> pd.DataFrame:
    """Format date columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with 'week_number' and 'year' columns.

    Returns:
    - pd.DataFrame: DataFrame with formatted date column.
    """
    df["week_number"] = df["week_number"].apply(lambda x: int(x))
    df["year"] = df["year"].apply(lambda x: int(x))
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + df["week_number"].astype(str) + "1", format="%Y%W%w"
    )
    # Find the closest Sunday for each date
    df["date"] = df["date"] + pd.to_timedelta(
        (6 - df["date"].dt.dayofweek) % 7, unit="D"
    )
    return df


def calculate_sample_size(
    population_size: int, confidence_level: float = 0.95, margin_of_error: float = 0.01
) -> int:
    """Calculate the required sample size for a given population.

    Parameters:
    - population_size (int): Total size of the population.
    - confidence_level (float): Confidence level for the calculation.
    - margin_of_error (float): Margin of error for the calculation.

    Returns:
    - int: Recommended sample size.
    """
    z_score = 1.96  # for a 95% confidence level
    p = 0.5  # assuming a conservative estimate for proportion
    numerator = z_score**2 * p * (1 - p)
    denominator = margin_of_error**2
    sample_size = math.ceil(
        (numerator / denominator) / (1 + ((numerator - 1) / population_size))
    )
    return sample_size


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load raw data from a CSV file.

    Parameters:
    - data_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the raw data.
    """
    return pd.read_csv(data_path)


def preprocess_data(df_raw: pd.DataFrame, sample_size: int) -> None:
    """Preprocess the raw data and save the results.

    Parameters:
    - df_raw (pd.DataFrame): Raw DataFrame containing commit history.
    - sample_size (int): Recommended sample size.
    """
    # ... (rest of the function remains unchanged)


if __name__ == "__main__":
    # Get the current working directory
    current_dir = os.getcwd()

    # Navigate to the data directory from the forecast directory
    data_path = os.path.join(current_dir, "../data/raw/commit_history_raw.csv")

    # Read the CSV file
    df_raw = load_raw_data(data_path)

    # Calculate the sample size
    total_repositories = len(
        df_raw.groupby(["repo_author_single", "year", "week_number"])[
            "commit_count"
        ].sum()
    )
    sample_size = calculate_sample_size(total_repositories)

    # Preprocess the data
    preprocess_data(df_raw, sample_size)
