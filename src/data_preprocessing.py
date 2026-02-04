import os
import pandas as pd
import numpy as np

# -----------------------------
# Path setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))

print("Original ratings shape:", ratings.shape)

# -----------------------------
# Cleaning
# -----------------------------
ratings.dropna(inplace=True)
ratings.drop_duplicates(inplace=True)
movies.dropna(inplace=True)
movies.drop_duplicates(inplace=True)

# -----------------------------
# FILTER: Active users & popular movies
# -----------------------------
user_counts = ratings["userId"].value_counts()
movie_counts = ratings["movieId"].value_counts()

active_users = user_counts[user_counts >= 50].index
popular_movies = movie_counts[movie_counts >= 100].index

ratings_filtered = ratings[
    ratings["userId"].isin(active_users) &
    ratings["movieId"].isin(popular_movies)
]

print("Filtered ratings shape:", ratings_filtered.shape)

# -----------------------------
# OPTIONAL: Further sampling (safe)
# -----------------------------
ratings_filtered = ratings_filtered.sample(n=500_000, random_state=42)

# -----------------------------
# Content-based prep
# -----------------------------
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

# -----------------------------
# Save cleaned data (NO matrix yet)
# -----------------------------
ratings_filtered.to_csv(
    os.path.join(DATA_DIR, "ratings_cleaned.csv"),
    index=False
)

movies.to_csv(
    os.path.join(DATA_DIR, "movies_cleaned.csv"),
    index=False
)

print("Preprocessing completed successfully!")

