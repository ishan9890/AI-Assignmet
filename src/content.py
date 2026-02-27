import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_movies(data_dir):
    return pd.read_csv(os.path.join(data_dir, "movies_cleaned.csv"))


def train_content_model(movies):
    movies = movies.copy()
    movies["genres"] = movies["genres"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity


def recommend(movie_title, movies, similarity, top_n=5):
    movies = movies.reset_index(drop=True)

    if movie_title not in movies["title"].values:
        return f"Movie '{movie_title}' not found in dataset."

    idx = movies[movies["title"] == movie_title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    indices = [i[0] for i in sim_scores]
    return movies.iloc[indices][["title", "genres"]].reset_index(drop=True)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    movies = load_movies(DATA_DIR)
    similarity = train_content_model(movies)

    recs = recommend("Toy Story (1995)", movies, similarity)
    print(recs)