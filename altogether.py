import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

def process_dataset(nrows, prefix, 
                    combined_data_path="/content/combined_data_1.txt", 
                    movies_path="/content/movies.csv", 
                    show_plot=False):
    """
    Process the raw ratings and movies data to produce a merged dataset,
    then map user and movie ids to contiguous indices, and save the outputs.
    
    Parameters:
      - nrows: Number of rows to read from the combined data file.
      - prefix: A string (e.g., "1M") used to name the output files.
      - combined_data_path: Path to the ratings file.
      - movies_path: Path to the movies (title/genre) file.
      - show_plot: Whether to display the rating distribution plot.
    
    Returns:
      merged_df: The merged and trimmed DataFrame.
      user_to_idx: Mapping dictionary for user ids.
      movie_to_idx: Mapping dictionary for movie ids.
    """
    
    print(f"\nProcessing {nrows} rows from combined data...")
    # -------------------------
    # Step 1: Load raw combined data (ratings file)
    # -------------------------
    df = pd.read_csv(combined_data_path, 
                     header=None, 
                     names=['Cust_Id', 'Rating'], 
                     usecols=[0, 1],
                     nrows=nrows)
    
    print("Initial raw data (first 5 rows):")
    print(df.head())
    print("Shape of raw data:", df.shape)
    
    # -------------------------
    # Step 2: Some basic counts and (optional) plot of rating distribution
    # -------------------------
    ratings_count = pd.DataFrame(df['Rating'].value_counts().sort_index(ascending=True))
    movie_count = df['Rating'].isnull().sum()  # these rows contain the movie id header
    customer_count = df['Cust_Id'].nunique() - movie_count
    Total_ratings_count = ratings_count.sum()[0]
    
    if show_plot:
        ax = ratings_count.plot(kind='barh', legend=False, figsize=(15, 10))
        plt.title(f"Total Pool: {movie_count} Movies, {customer_count} Customers, {Total_ratings_count} Ratings", fontsize=10)
        plt.xlabel("Count Of Rating")
        plt.ylabel("Rating")
        for i in ax.patches:
            percentage = i.get_width() / Total_ratings_count * 100
            ax.text(i.get_width(), i.get_y() + 0.1, f"{round(percentage,0)}%", fontsize=10, color='black')
        plt.ticklabel_format(style='plain', axis='x')
        plt.show()
    
    # -------------------------
    # Step 3: Process rows with NaN ratings to generate Movie IDs
    # -------------------------
    df_nan = df[df['Rating'].isnull()].copy()
    df_nan.reset_index(inplace=True)
    
    movie_np = np.array([])
    movie_id = 1
    # Loop over consecutive pairs of indices in df_nan to assign movie ids
    for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
        temp = np.full((i - j - 1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1
        
    # Handle the last block of ratings after the final movie header row
    if not df_nan.empty:
        last_record = np.full((len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
        movie_np = np.append(movie_np, last_record)
    
    print(f"Generated movie ID array with length: {len(movie_np)}")
    
    # Remove the movie header rows (which have NaN in 'Rating')
    df = df[pd.notnull(df['Rating'])].copy()
    # Add the computed Movie_Id column and convert Cust_Id to integer
    df['Movie_Id'] = movie_np.astype(int)
    df['Cust_Id'] = df['Cust_Id'].astype(int)
    
    print("Data after adding Movie_Id (first 10 rows):")
    print(df.head(10))
    
    # -------------------------
    # Step 4: Load the movies (title and genres) data and process genres
    # -------------------------
    df_title = pd.read_csv(movies_path, header=None, names=['Movie_Id', 'Title', 'Genres'], on_bad_lines='skip')
    df_title['Genres'] = df_title['Genres'].str.split('|')
    expanded_data = df_title.explode('Genres')
    # Remove irrelevant genres
    expanded_data = expanded_data[(expanded_data['Genres'] != 'IMAX') & (expanded_data['Genres'] != '(no genres listed)')]
    
    # Ensure Movie_Id is numeric in both dataframes
    df['Movie_Id'] = pd.to_numeric(df['Movie_Id'], errors='coerce')
    expanded_data['Movie_Id'] = pd.to_numeric(expanded_data['Movie_Id'], errors='coerce')
    
    # -------------------------
    # Step 5: Merge the ratings data with the movies data on Movie_Id
    # -------------------------
    merged_df = pd.merge(df, expanded_data, on='Movie_Id', how='inner')
    print("Merged data (first 5 rows):")
    print(merged_df.head())
    print("Merged data shape:", merged_df.shape)
    
    # -------------------------
    # Step 6: Data Cleaning – remove movies and users with very few ratings
    # -------------------------
    df_movie_summary = merged_df.groupby('Movie_Id')['Rating'].agg(['count', 'mean'])
    movie_benchmark = round(df_movie_summary['count'].quantile(0.7), 0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index
    
    df_cust_summary = merged_df.groupby('Cust_Id')['Rating'].agg(['count', 'mean'])
    cust_benchmark = round(df_cust_summary['count'].quantile(0.7), 0)
    drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index
    
    print(f"Before trimming, merged data shape: {merged_df.shape}")
    merged_df = merged_df[~merged_df['Movie_Id'].isin(drop_movie_list)]
    merged_df = merged_df[~merged_df['Cust_Id'].isin(drop_cust_list)]
    print(f"After trimming, merged data shape: {merged_df.shape}")
    
    # -------------------------
    # Step 7: Mapping – assign contiguous indices to users and movies
    # -------------------------
    unique_users = merged_df['Cust_Id'].unique()
    unique_movies = merged_df['Movie_Id'].unique()
    
    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    movie_to_idx = {mid: j for j, mid in enumerate(unique_movies)}
    
    merged_df["user_idx"] = merged_df["Cust_Id"].map(user_to_idx)
    merged_df["movie_idx"] = merged_df["Movie_Id"].map(movie_to_idx)
    
    df_mapped = merged_df[["user_idx", "movie_idx", "Rating"]].copy()
    
    # -------------------------
    # Step 8: Save the mapped dataset and the mapping dictionaries to disk
    # -------------------------
    mapped_csv_filename = f"mapped_merged_data_{prefix}.csv"
    user_map_filename = f"user_map_{prefix}.pkl"
    movie_map_filename = f"movie_map_{prefix}.pkl"
    
    df_mapped.to_csv(mapped_csv_filename, index=False)
    with open(user_map_filename, "wb") as f:
        pickle.dump(user_to_idx, f)
    with open(movie_map_filename, "wb") as f:
        pickle.dump(movie_to_idx, f)
    
    print(f"Saved mapped data to: {mapped_csv_filename}")
    print(f"Saved user map to: {user_map_filename}")
    print(f"Saved movie map to: {movie_map_filename}")
    
    # Optionally, return the results for further inspection
    return merged_df, user_to_idx, movie_to_idx

# -------------------------
# Main Loop: Process for 1M, 2M, 4M, 6M, 8M, 16M, 32M rows
# -------------------------
sizes = [1, 2, 4, 6, 8, 16, 32]

for size in sizes:
    nrows = size * 10**6  # e.g., 1M rows = 1,000,000 rows
    prefix = f"{size}M"
    print("\n" + "="*50)
    print(f"Processing dataset with {prefix} rows")
    merged_df, user_map, movie_map = process_dataset(nrows, prefix, show_plot=False)
