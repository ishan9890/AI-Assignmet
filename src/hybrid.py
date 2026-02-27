import pandas as pd
from collaborative import build_matrix, train_model
from content import train_content_model, recommend


def hybrid_recommend(user_id, ratings, movies):
    matrix = build_matrix(ratings)
    matrix = matrix.fillna(0)

    if user_id not in matrix.index:
        return f"User ID {user_id} not found in ratings data."

    model = train_model(matrix, k=5)

    user_vector = matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vector)

    neighbor_ids = matrix.index[indices.flatten()][1:]
    collab_scores = matrix.loc[neighbor_ids].mean()

    top_movie_id = collab_scores.sort_values(ascending=False).index[0]

    movie_row = movies[movies["movieId"] == top_movie_id]
    if movie_row.empty:
        return f"Could not find movie with ID {top_movie_id}."

    top_movie_title = movie_row["title"].values[0]

    movies = movies.reset_index(drop=True)
    similarity = train_content_model(movies)
    return recommend(top_movie_title, movies, similarity)