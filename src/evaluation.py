import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt

def load_data(data_dir):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings_cleaned.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies_cleaned.csv"))
    return ratings, movies


def split_data(ratings):
    return train_test_split(ratings, test_size=0.3, random_state=42)


def build_user_matrix(ratings):
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0
    )

def train_collaborative(matrix, k=5):
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k)
    model.fit(matrix)
    return model

def predict_collaborative(model, matrix, user, movie):
    if user not in matrix.index or movie not in matrix.columns:
        return None

    user_vector = matrix.loc[user].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vector)
    neighbors = matrix.index[indices.flatten()]
    return matrix.loc[neighbors, movie].mean()

def train_content(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

def predict_content(user, movie, ratings, movies, similarity):
    user_movies = ratings[ratings["userId"] == user]
    if user_movies.empty:
        return None

    weighted_sum = 0
    sim_sum = 0

    for _, row in user_movies.iterrows():
        rated_movie = row["movieId"]
        rating = row["rating"]

        idx1 = movies[movies["movieId"] == movie].index
        idx2 = movies[movies["movieId"] == rated_movie].index

        if len(idx1) == 0 or len(idx2) == 0:
            continue

        sim = similarity[idx1[0], idx2[0]]
        weighted_sum += sim * rating
        sim_sum += sim

    if sim_sum == 0:
        return None

    return weighted_sum / sim_sum


def predict_hybrid(collab_pred, content_pred, alpha=0.5):
    if collab_pred is None and content_pred is None:
        return None
    if collab_pred is None:
        return content_pred
    if content_pred is None:
        return collab_pred

    return alpha * collab_pred + (1 - alpha) * content_pred


def evaluate_models(train, test, movies):
    matrix = build_user_matrix(train)
    collab_model = train_collaborative(matrix)
    similarity = train_content(movies)

    collab_preds, content_preds, hybrid_preds = [], [], []
    actuals = []

    for _, row in test.iterrows():
        user = row["userId"]
        movie = row["movieId"]
        rating = row["rating"]

        collab = predict_collaborative(collab_model, matrix, user, movie)
        content = predict_content(user, movie, train, movies, similarity)
        hybrid = predict_hybrid(collab, content)

        if hybrid is not None:
            collab_preds.append(collab if collab else 0)
            content_preds.append(content if content else 0)
            hybrid_preds.append(hybrid)
            actuals.append(rating)

    results = {}

    # Collaborative
    results["Collaborative RMSE"] = sqrt(mean_squared_error(actuals, collab_preds))
    results["Collaborative MAE"] = mean_absolute_error(actuals, collab_preds)

    # Content
    results["Content RMSE"] = sqrt(mean_squared_error(actuals, content_preds))
    results["Content MAE"] = mean_absolute_error(actuals, content_preds)

    # Hybrid
    results["Hybrid RMSE"] = sqrt(mean_squared_error(actuals, hybrid_preds))
    results["Hybrid MAE"] = mean_absolute_error(actuals, hybrid_preds)

    return results


# ---------------------------
# Run Evaluation
# ---------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    ratings, movies = load_data(DATA_DIR)
    train, test = split_data(ratings)

    results = evaluate_models(train, test, movies)

    print("\nMODEL COMPARISON RESULTS")
    print("------------------------")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")