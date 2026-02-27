import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ==============================
# 1. LOAD DATA
# ==============================
def load_data(data_dir):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    return ratings, movies


# ==============================
# 2. INITIAL INSPECTION (For Report)
# ==============================
def inspect_data(ratings, movies):
    print("Ratings Dataset Shape:", ratings.shape)
    print("Movies Dataset Shape:", movies.shape)

    print("\nRatings Info:")
    print(ratings.info())

    print("\nMovies Info:")
    print(movies.info())

    print("\nMissing Values in Ratings:")
    print(ratings.isnull().sum())

    print("\nMissing Values in Movies:")
    print(movies.isnull().sum())


# ==============================
# 3. CLEAN DATA
# ==============================
def clean_data(ratings, movies):
    ratings = ratings.dropna().drop_duplicates()
    movies = movies.dropna().drop_duplicates()
    return ratings, movies


# ==============================
# 4. FILTER DATA (Reduce Size)
# ==============================
def filter_data(ratings, min_user=50, min_movie=100):
    user_counts = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()

    active_users = user_counts[user_counts >= min_user].index
    popular_movies = movie_counts[movie_counts >= min_movie].index

    filtered_ratings = ratings[
        ratings["userId"].isin(active_users) &
        ratings["movieId"].isin(popular_movies)
    ]

    print("\nFiltered Ratings Shape:", filtered_ratings.shape)

    return filtered_ratings


# ==============================
# 5. CREATE USER-MOVIE MATRIX
# ==============================
def create_user_movie_matrix(ratings):
    user_movie_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    )

    print("\nUser-Movie Matrix Shape:", user_movie_matrix.shape)
    print(user_movie_matrix.head())

    return user_movie_matrix


# ==============================
# 6. TRAIN-TEST SPLIT (70/30)
# ==============================
def split_data(ratings):
    train_data, test_data = train_test_split(
        ratings,
        test_size=0.3,
        random_state=42
    )

    print("\nTraining Set Size:", len(train_data))
    print("Testing Set Size:", len(test_data))

    return train_data, test_data


# ==============================
# 7. SAVE DATA
# ==============================
def save_data(ratings, movies, user_movie_matrix, train_data, test_data, data_dir):
    ratings.to_csv(os.path.join(data_dir, "ratings_cleaned.csv"), index=False)
    movies.to_csv(os.path.join(data_dir, "movies_cleaned.csv"), index=False)
    user_movie_matrix.to_csv(os.path.join(data_dir, "user_movie_matrix.csv"))
    train_data.to_csv(os.path.join(data_dir, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(data_dir, "test_data.csv"), index=False)


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # 1. Load
    ratings, movies = load_data(DATA_DIR)

    # 2. Inspect (FOR REPORT FIGURE)
    inspect_data(ratings, movies)

    # 3. Clean
    ratings, movies = clean_data(ratings, movies)

    # 4. Filter (Reduce heavy memory usage)
    ratings = filter_data(ratings)

    # 5. Optional Sampling (TEMPORARY - REMOVE if dataset small)
    # This prevents memory crash
    if len(ratings) > 20000:
        ratings = ratings.sample(20000, random_state=42)
        print("\nSampled Ratings Shape:", ratings.shape)

    # 6. Create Matrix
    user_movie_matrix = create_user_movie_matrix(ratings)

    # 7. Train/Test Split
    train_data, test_data = split_data(ratings)

    # 8. Save
    save_data(ratings, movies, user_movie_matrix, train_data, test_data, DATA_DIR)

    print("\nPreprocessing completed successfully.")