# How to Use the Movie Recommender on Kaggle

This guide explains how to use the `MovieRecommender` class in a Kaggle Notebook environment.

## Step 1: Create a New Notebook
1. Go to [Kaggle](https://www.kaggle.com/).
2. Click on **"Create"** -> **"New Notebook"**.

## Step 2: Add the Dataset
1. In the notebook sidebar (right side), click on **"Add Input"**.
2. Search for **"MovieLens"** (the dataset used in this project is typically `movielens/ratings.csv` and `movielens/movies.csv`, often found in datasets like "MovieLens 20M Dataset" or "MovieLens Small Latest Dataset").
3. Click the **"+"** button to add it to your notebook.
4. Once added, you will see the files under `/kaggle/input/`.

## Step 3: Copy the Code
Copy the entire content of `recommendation_system_improved.py` into the first code cell of your notebook.

## Step 4: Run the Recommender
In a new cell, instantiate and run the recommender using the correct paths.

> **Note:** The paths below (`/kaggle/input/...`) depend on the specific dataset you added. You can copy the exact path by clicking the "Copy file path" icon next to the file in the sidebar.

```python
# 1. Instantiate the recommender
# You can adjust min_vote_threshold based on the dataset size (e.g., 50 for small, 500 for large)
recommender = MovieRecommender(min_vote_threshold=50)

# 2. Define your paths (Example paths, verify yours!)
# If you used the standard MovieLens dataset, paths might look like this:
RATINGS_PATH = "/kaggle/input/movielens-small/ratings.csv"
MOVIES_PATH = "/kaggle/input/movielens-small/movies.csv"

# 3. Load and Train
try:
    recommender.load_data(RATINGS_PATH, MOVIES_PATH)
    recommender.preprocess()
    recommender.train()

    # 4. Get Recommendations
    print("\n--- Movies similar to 'Toy Story (1995)' ---")
    print(recommender.recommend_similar_items("Toy Story (1995)"))

    print("\n--- Recommendations for User ID 1 ---")
    print(recommender.recommend_for_user(user_id=1))

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please verify the file paths in the 'Input' section of the sidebar.")
except Exception as e:
    print(f"An error occurred: {e}")
```

## Tips
*   **Memory:** The code is optimized with sparse matrices, so it should handle larger datasets (like MovieLens 20M) better than standard pandas pivot tables.
*   **Thresholds:** If the training is too slow or memory intensive, increase `min_vote_threshold` to ignore less popular movies.
