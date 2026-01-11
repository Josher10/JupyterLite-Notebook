import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional, Tuple
import os

class MovieRecommender:
    """
    A robust movie recommendation system using Item-Based Collaborative Filtering
    with K-Nearest Neighbors on sparse matrices.
    """

    def __init__(self, min_vote_threshold: int = 50, n_neighbors: int = 20):
        self.min_vote_threshold = min_vote_threshold
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
        self.movies_df = None
        self.ratings_df = None
        self.movie_user_matrix_sparse = None
        self.matrix_index = None  # To map titles to matrix indices

    def load_data(self, ratings_path: str, movies_path: str):
        """Loads and merges data."""
        if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
            raise FileNotFoundError("Data files not found.")

        self.ratings_df = pd.read_csv(ratings_path, dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        self.movies_df = pd.read_csv(movies_path, dtype={'movieId': 'int32', 'title': 'str'})

        # Merge to check data integrity early
        print(f"Loaded {len(self.ratings_df)} ratings and {len(self.movies_df)} movies.")

    def preprocess(self):
        """Filters data and creates the sparse matrix."""
        if self.ratings_df is None or self.movies_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # 1. Filter movies with too few votes to ensure statistical significance
        movie_counts = self.ratings_df.groupby('movieId')['rating'].count()
        popular_movies = movie_counts[movie_counts >= self.min_vote_threshold].index

        filtered_ratings = self.ratings_df[self.ratings_df['movieId'].isin(popular_movies)]
        print(f"Filtered down to {len(popular_movies)} movies with > {self.min_vote_threshold} votes.")

        # 2. Merge with titles for matrix indexing
        df_merged = filtered_ratings.merge(self.movies_df, on='movieId')

        # 3. Create Pivot Table (Movies x Users)
        # Note: We use Title as index for readability, but movieId is safer for duplicates.
        # Here we assume titles are unique or we want to aggregate by title.
        self.pivot_df = df_merged.pivot_table(index='title', columns='userId', values='rating').fillna(0)

        # 4. Convert to Sparse Matrix for memory efficiency
        self.movie_user_matrix_sparse = csr_matrix(self.pivot_df.values)
        self.matrix_index = self.pivot_df.index

    def train(self):
        """Fits the KNN model."""
        if self.movie_user_matrix_sparse is None:
            raise ValueError("Data not preprocessed.")

        self.model.fit(self.movie_user_matrix_sparse)
        print("Model trained successfully.")

    def recommend(self, movie_title: str, n_recommendations: int = 5) -> pd.DataFrame:
        """
        Recommends movies similar to the given movie title.
        """
        if movie_title not in self.matrix_index:
            # Simple fuzzy search fallback could be implemented here
            return pd.DataFrame({'Error': [f"Movie '{movie_title}' not found in the catalog."]})

        # Get index
        query_index = self.matrix_index.get_loc(movie_title)

        # Find neighbors
        distances, indices = self.model.kneighbors(
            self.pivot_df.iloc[query_index, :].values.reshape(1, -1),
            n_neighbors=n_recommendations + 1
        )

        recs = []
        # Skip the first one because it is the movie itself
        for i in range(1, len(distances.flatten())):
            idx = indices.flatten()[i]
            dist = distances.flatten()[i]
            title = self.matrix_index[idx]

            recs.append({
                'Title': title,
                'Similarity': 1 - dist, # Cosine similarity = 1 - Cosine distance
                'Distance': dist
            })

        return pd.DataFrame(recs)

# Example usage (commented out as data is missing)
# if __name__ == "__main__":
#     recommender = MovieRecommender()
#     recommender.load_data("ratings.csv", "movies.csv")
#     recommender.preprocess()
#     recommender.train()
#     print(recommender.recommend("Matrix, The (1999)"))
