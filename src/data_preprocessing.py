import os
import pandas as pd
import numpy as np


def load_data(data_dir):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    return ratings, movies

def clean_data(ratings, movies):
    ratings = ratings.dropna().drop_duplicates()
    movies = movies.dropna().drop_duplicates()
    return ratings, movies

def filter_data(ratings, min_user_ratings=50, min_movie_ratings=100):
    user_counts = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()

    active_users = user_counts[user_counts >= min_user_ratings].index
    popular_movies = movie_counts[movie_counts >= min_movie_ratings].index

    filtered_ratings = ratings[
        ratings["userId"].isin(active_users) &
        ratings["movieId"].isin(popular_movies)
    ]

    return filtered_ratings



def preprocess_movies(movies):
    movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)
    return movies



def save_data(ratings, movies, data_dir):
    ratings.to_csv(os.path.join(data_dir, "ratings_cleaned.csv"), index=False)
    movies.to_csv(os.path.join(data_dir, "movies_cleaned.csv"), index=False)


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    print("Loading datasets...")
    ratings, movies = load_data(DATA_DIR)

    print("Original ratings shape:", ratings.shape)
    print("Original movies shape:", movies.shape)

    print("\nCleaning data...")
    ratings, movies = clean_data(ratings, movies)

    print("Filtering ratings...")
    ratings = filter_data(ratings)

    ratings = ratings.sample(n=500_000, random_state=42)

    print("Filtered ratings shape:", ratings.shape)

    print("Preprocessing movie content...")
    movies = preprocess_movies(movies)

    print("Saving cleaned datasets...")
    save_data(ratings, movies, DATA_DIR)

    print("\nData preprocessing completed successfully!")
