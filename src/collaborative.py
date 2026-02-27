import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from math import sqrt


def load_data(data_dir):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings_cleaned.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies_cleaned.csv"))
    return ratings, movies


def split_data(ratings):
    train, test = train_test_split(
        ratings,
        test_size=0.3,
        random_state=42
    )
    return train, test


def build_matrix(ratings):
    matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )
    return matrix


def train_model(user_matrix, k=5):
    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=k + 1
    )
    model.fit(user_matrix.fillna(0))
    return model


def evaluate_rmse(model, train_matrix, test_data, k=5):
    predictions = []
    actuals = []

    train_filled = train_matrix.fillna(0)

    for _, row in test_data.iterrows():
        user = row["userId"]
        movie = row["movieId"]
        rating = row["rating"]

        if user in train_matrix.index and movie in train_matrix.columns:
            user_vector = train_filled.loc[user].values.reshape(1, -1)
            distances, indices = model.kneighbors(user_vector)

            neighbor_ids = train_matrix.index[indices.flatten()][1:k + 1]
            neighbor_ratings = train_matrix.loc[neighbor_ids, movie].dropna()

            if len(neighbor_ratings) > 0:
                predicted_rating = neighbor_ratings.mean()
                predictions.append(predicted_rating)
                actuals.append(rating)

    if len(predictions) == 0:
        print("No valid predictions made.")
        return None

    rmse = sqrt(mean_squared_error(actuals, predictions))
    return rmse


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    ratings, movies = load_data(DATA_DIR)
    train_data, test_data = split_data(ratings)

    print("Train size:", len(train_data))
    print("Test size:", len(test_data))

    train_matrix = build_matrix(train_data)

    print("User-Movie Matrix Shape:", train_matrix.shape)
    print(train_matrix.head())

    model = train_model(train_matrix, k=5)
    rmse = evaluate_rmse(model, train_matrix, test_data, k=5)

    print("\nCollaborative Filtering RMSE:", rmse)