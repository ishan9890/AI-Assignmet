import os
import pandas as pd
from hybrid import hybrid_recommend


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings_cleaned.csv"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies_cleaned.csv"))

    user_id = int(input("Enter User ID: "))
    recommendations = hybrid_recommend(user_id, ratings, movies)

    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print("\nRecommended Movies:")
        print(recommendations)