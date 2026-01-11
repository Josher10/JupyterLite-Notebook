import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional, Tuple, Union
import os

class MovieRecommender:
    """
    A robust movie recommendation system using Item-Based Collaborative Filtering
    with K-Nearest Neighbors on sparse matrices.

    Improved for memory efficiency by avoiding dense pivot tables.
    """

    def __init__(self, min_vote_threshold: int = 50, n_neighbors: int = 20):
        self.min_vote_threshold = min_vote_threshold
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
        self.movies_df = None
        self.ratings_df = None
        self.movie_user_matrix_sparse = None

        # Mappings
        self.movie_id_to_idx = {} # movieId -> matrix row index
        self.idx_to_movie_id = {} # matrix row index -> movieId
        self.movie_id_to_title = {} # movieId -> Title

    def load_data(self, ratings_path: str, movies_path: str):
        """Loads and merges data."""
        if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
            raise FileNotFoundError(f"Data files not found at {ratings_path} or {movies_path}")

        self.ratings_df = pd.read_csv(ratings_path, dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        self.movies_df = pd.read_csv(movies_path, dtype={'movieId': 'int32', 'title': 'str'})

        # Create title mapping
        self.movie_id_to_title = self.movies_df.set_index('movieId')['title'].to_dict()
        print(f"Loaded {len(self.ratings_df)} ratings and {len(self.movies_df)} movies.")

    def preprocess(self):
        """Filters data and creates the sparse matrix directly."""
        if self.ratings_df is None or self.movies_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # 1. Filter movies with too few votes
        movie_counts = self.ratings_df.groupby('movieId')['rating'].count()
        popular_movies = movie_counts[movie_counts >= self.min_vote_threshold].index

        filtered_ratings = self.ratings_df[self.ratings_df['movieId'].isin(popular_movies)].copy()
        print(f"Filtered down to {len(popular_movies)} movies with >= {self.min_vote_threshold} votes.")

        # 2. Create Mappings for Sparse Matrix
        # Unique users and movies in the filtered dataset
        unique_movies = filtered_ratings['movieId'].unique()
        unique_users = filtered_ratings['userId'].unique()

        # Map IDs to continuous indices (0 to N-1)
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}

        user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}

        # Apply mapping to dataframe
        filtered_ratings['movie_idx'] = filtered_ratings['movieId'].map(self.movie_id_to_idx)
        filtered_ratings['user_idx'] = filtered_ratings['userId'].map(user_id_to_idx)

        # 3. Create Sparse Matrix (Rows=Movies, Cols=Users)
        # Using coordinate format to build CSR
        num_movies = len(unique_movies)
        num_users = len(unique_users)

        self.movie_user_matrix_sparse = csr_matrix(
            (filtered_ratings['rating'], (filtered_ratings['movie_idx'], filtered_ratings['user_idx'])),
            shape=(num_movies, num_users)
        )

        print(f"Sparse Matrix Created. Shape: {self.movie_user_matrix_sparse.shape}")

    def train(self):
        """Fits the KNN model."""
        if self.movie_user_matrix_sparse is None:
            raise ValueError("Data not preprocessed.")

        self.model.fit(self.movie_user_matrix_sparse)
        print("Model trained successfully.")

    def _get_movie_id_by_title(self, title: str) -> Optional[int]:
        """Helper to find movieId by partial title match."""
        # Exact match first
        for mid, t in self.movie_id_to_title.items():
            if t == title:
                return mid
        # Fuzzy match (case insensitive)
        title_lower = title.lower()
        for mid, t in self.movie_id_to_title.items():
            if isinstance(t, str) and title_lower in t.lower():
                return mid
        return None

    def recommend_similar_items(self, movie_title: str, n_recommendations: int = 5) -> pd.DataFrame:
        """
        Recommends movies similar to the given movie title (Item-Item).
        """
        movie_id = self._get_movie_id_by_title(movie_title)

        if movie_id is None:
             return pd.DataFrame({'Error': [f"Movie '{movie_title}' not found in catalog."]})

        if movie_id not in self.movie_id_to_idx:
            return pd.DataFrame({'Error': [f"Movie '{movie_title}' (ID {movie_id}) was excluded by popularity filter."]})

        # Get matrix row index
        query_idx = self.movie_id_to_idx[movie_id]

        # Find neighbors
        # Note: We pass the vector for the movie from the sparse matrix
        distances, indices = self.model.kneighbors(
            self.movie_user_matrix_sparse[query_idx],
            n_neighbors=n_recommendations + 1
        )

        recs = []
        for i in range(1, len(distances.flatten())):
            neighbor_idx = indices.flatten()[i]
            dist = distances.flatten()[i]

            # Map back index -> movieId -> Title
            rec_movie_id = self.idx_to_movie_id[neighbor_idx]
            rec_title = self.movie_id_to_title.get(rec_movie_id, f"Unknown ID {rec_movie_id}")

            recs.append({
                'Title': rec_title,
                'Similarity': 1 - dist,
                'Distance': dist
            })

        return pd.DataFrame(recs)

    def recommend_for_user(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """
        Generates personalized recommendations for a user based on their high-rated movies.
        Aggregates similar items to the ones the user liked.
        """
        if self.ratings_df is None:
             raise ValueError("Data not loaded.")

        # 1. Get User Profile
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]

        if user_ratings.empty:
            return pd.DataFrame({'Error': [f"User {user_id} has no ratings."]})

        # Filter for liked movies (e.g., rating > 3.0)
        liked_movies_df = user_ratings[user_ratings['rating'] > 3.0]

        if liked_movies_df.empty:
             return pd.DataFrame({'Error': [f"User {user_id} has no highly rated movies (> 3.0)."]})

        # 2. Aggregate Candidates
        candidates = {} # MovieID -> Sum of Weighted Similarity Scores

        # Only consider movies that exist in our trained matrix
        valid_liked_movies = liked_movies_df[liked_movies_df['movieId'].isin(self.movie_id_to_idx)]

        if valid_liked_movies.empty:
            return pd.DataFrame({'Error': ["None of the user's liked movies are in the popular set."]})

        print(f"Generating recommendations based on {len(valid_liked_movies)} liked movies...")

        for _, row in valid_liked_movies.iterrows():
            movie_id = row['movieId']
            rating = row['rating']

            # Get matrix index
            query_idx = self.movie_id_to_idx[movie_id]

            distances, indices = self.model.kneighbors(
                self.movie_user_matrix_sparse[query_idx],
                n_neighbors=self.n_neighbors + 1
            )

            for i in range(1, len(distances.flatten())):
                neighbor_idx = indices.flatten()[i]
                dist = distances.flatten()[i]

                # Convert back to Movie ID
                neighbor_movie_id = self.idx_to_movie_id[neighbor_idx]

                # Similarity (1 - cosine distance)
                similarity = 1 - dist

                # Weight by user's rating
                weighted_score = similarity * rating

                if neighbor_movie_id in candidates:
                    candidates[neighbor_movie_id] += weighted_score
                else:
                    candidates[neighbor_movie_id] = weighted_score

        # 3. Filter Watched and Sort
        watched_ids = set(user_ratings['movieId'])

        final_recs = []
        for mid, score in candidates.items():
            if mid not in watched_ids:
                final_recs.append({
                    'Title': self.movie_id_to_title.get(mid, f"ID {mid}"),
                    'Score': score
                })

        if not final_recs:
             return pd.DataFrame(columns=['Title', 'Score'])

        recs_df = pd.DataFrame(final_recs).sort_values(by='Score', ascending=False).head(n_recommendations)
        return recs_df

# Usage Example
if __name__ == "__main__":
    # Example paths - replace with your actual file paths
    # RATINGS_FILE = "ratings.csv"
    # MOVIES_FILE = "movies.csv"

    # try:
    #     recommender = MovieRecommender(min_vote_threshold=50)
    #     recommender.load_data(RATINGS_FILE, MOVIES_FILE)
    #     recommender.preprocess()
    #     recommender.train()
    #
    #     print("\n--- Similar to 'Matrix, The' ---")
    #     print(recommender.recommend_similar_items("Matrix, The"))
    #
    #     print("\n--- Recommendations for User 1 ---")
    #     print(recommender.recommend_for_user(user_id=1))
    # except Exception as e:
    #     print(f"Execution skipped: {e}")
    pass
