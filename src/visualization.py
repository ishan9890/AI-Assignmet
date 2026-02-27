import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np



def plot_predicted_vs_actual(actual, predicted):
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(actual, predicted, alpha=0.5, color="#2E75B6", edgecolors="white",
               linewidth=0.5, s=60, label="Predictions")

    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Prediction")

    ax.set_xlabel("Actual Ratings", fontsize=12)
    ax.set_ylabel("Predicted Ratings", fontsize=12)
    ax.set_title("Predicted vs Actual Ratings", fontsize=14, fontweight="bold", pad=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig("predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.show()



def plot_model_comparison():
    
    results = {
        "Collaborative RMSE": 3.5669,
        "Collaborative MAE":  3.4172,
        "Content RMSE":       2.3421,
        "Content MAE":        1.8164,
        "Hybrid RMSE":        2.6191,
        "Hybrid MAE":         2.2832,
    }

    models = ["Collaborative\nFiltering", "Content-Based\nFiltering", "Hybrid\nModel"]
    rmse_vals = [results["Collaborative RMSE"], results["Content RMSE"], results["Hybrid RMSE"]]
    mae_vals  = [results["Collaborative MAE"],  results["Content MAE"],  results["Hybrid MAE"]]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(x - width / 2, rmse_vals, width, label="RMSE",
                   color="#1F3864", alpha=0.88, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, mae_vals,  width, label="MAE",
                   color="#2E75B6", alpha=0.88, edgecolor="white", linewidth=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Model", fontsize=12, labelpad=8)
    ax.set_ylabel("Error Value", fontsize=12, labelpad=8)
    ax.set_title("Model Comparison – RMSE & MAE", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, max(rmse_vals) * 1.2)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_rating_distribution(data_dir):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings_cleaned.csv"))

    fig, ax = plt.subplots(figsize=(8, 5))

    rating_counts = ratings["rating"].value_counts().sort_index()
    bars = ax.bar(rating_counts.index, rating_counts.values,
                  width=0.35, color="#2E75B6", alpha=0.85,
                  edgecolor="white", linewidth=0.8)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Rating", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of Ratings", fontsize=12, labelpad=8)
    ax.set_title("Rating Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(sorted(ratings["rating"].unique()))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    mean_r = ratings["rating"].mean()
    median_r = ratings["rating"].median()
    ax.axvline(mean_r, color="red", linestyle="--", linewidth=1.2, label=f"Mean: {mean_r:.2f}")
    ax.axvline(median_r, color="orange", linestyle="--", linewidth=1.2, label=f"Median: {median_r:.1f}")
    ax.legend(fontsize=10)

    fig.tight_layout()
    plt.savefig("rating_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_top_movies(data_dir):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings_cleaned.csv"))
    movies  = pd.read_csv(os.path.join(data_dir, "movies_cleaned.csv"))

    # Require at least 50 ratings to be eligible (avoids obscure 1-rating films)
    movie_counts = ratings.groupby("movieId")["rating"].count()
    popular_ids  = movie_counts[movie_counts >= 50].index

    top_movies = (
        ratings[ratings["movieId"].isin(popular_ids)]
        .groupby("movieId")["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    top_movies = top_movies.merge(movies[["movieId", "title"]], on="movieId")


    top_movies["short_title"] = top_movies["title"].apply(
        lambda t: t[:40] + "..." if len(t) > 40 else t
    )

    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(top_movies)))[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_movies["short_title"], top_movies["rating"],
                   color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, top_movies["rating"]):
        ax.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="right",
                fontsize=9, fontweight="bold", color="white")

    ax.set_xlabel("Average Rating", fontsize=12, labelpad=8)
    ax.set_title("Top 10 Highest Rated Movies\n(minimum 50 ratings)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 5.3)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig("top_movies.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_ratings_per_user(data_dir):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings_cleaned.csv"))

    user_counts = ratings.groupby("userId")["rating"].count()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(user_counts, bins=40, color="#1F3864", alpha=0.85, edgecolor="white", linewidth=0.6)

    ax.axvline(user_counts.mean(), color="red", linestyle="--", linewidth=1.3,
               label=f"Mean: {user_counts.mean():.0f} ratings")
    ax.axvline(user_counts.median(), color="orange", linestyle="--", linewidth=1.3,
               label=f"Median: {user_counts.median():.0f} ratings")

    ax.set_xlabel("Number of Ratings per User", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of Users", fontsize=12, labelpad=8)
    ax.set_title("Ratings per User Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig("ratings_per_user.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_ratings_per_movie(data_dir):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings_cleaned.csv"))

    movie_counts = ratings.groupby("movieId")["rating"].count()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(movie_counts, bins=40, color="#2E75B6", alpha=0.85, edgecolor="white", linewidth=0.6)

    ax.axvline(movie_counts.mean(), color="red", linestyle="--", linewidth=1.3,
               label=f"Mean: {movie_counts.mean():.0f} ratings")
    ax.axvline(movie_counts.median(), color="orange", linestyle="--", linewidth=1.3,
               label=f"Median: {movie_counts.median():.0f} ratings")

    ax.set_xlabel("Number of Ratings per Movie", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of Movies", fontsize=12, labelpad=8)
    ax.set_title("Ratings per Movie Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig("ratings_per_movie.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_genre_distribution(data_dir):
    movies = pd.read_csv(os.path.join(data_dir, "movies_cleaned.csv"))


    genre_series = movies["genres"].dropna().str.split("|").explode()
    genre_counts  = genre_series.value_counts().head(15)

    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(genre_counts)))[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(genre_counts.index[::-1], genre_counts.values[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, genre_counts.values[::-1]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=9)

    ax.set_xlabel("Number of Movies", fontsize=12, labelpad=8)
    ax.set_title("Top 15 Movie Genres in Dataset", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig("genre_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    print("1. Generating Model Comparison Plot...")
    plot_model_comparison()

    print("2. Generating Rating Distribution Plot...")
    plot_rating_distribution(DATA_DIR)

    print("3. Generating Top 10 Highest Rated Movies Plot...")
    plot_top_movies(DATA_DIR)

    print("4. Generating Ratings per User Distribution...")
    plot_ratings_per_user(DATA_DIR)

    print("5. Generating Ratings per Movie Distribution...")
    plot_ratings_per_movie(DATA_DIR)

    print("6. Generating Genre Distribution Plot...")
    plot_genre_distribution(DATA_DIR)

    print("\nAll visualizations completed and saved as PNG files.")