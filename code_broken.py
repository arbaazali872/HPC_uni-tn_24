#Import the libararies required for model building.
from scipy.sparse import csr_matrix
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate,train_test_split

# Help us to read the dataset for SVD algorithm
reader=Reader()

# This line creates a Surprise Dataset object ('surprise_data') containing the data from the DataFrame

surprise_data=Dataset.load_from_df(merged_df[['Cust_Id', 'Movie_Id', 'Rating']], reader)

# Creating an instance of the SVD algorithm
# This initializes the SVD algorithm, which will be used for collaborative filtering-based recommendation

svd=SVD()

# Performing cross-validation on the Surprise Dataset object (surprise_data) using the SVD algorithm
# This evaluates the performance of the SVD algorithm using 3-fold cross-validation
# 'RMSE' (Root Mean Squared Error) and 'MAE' (Mean Absolute Error) are the evaluation metrics used
# The 'cv' parameter specifies the number of folds for cross-validation, which is set to 3
# The 'verbose' parameter controls whether to print progress messages during cross-validation

cross_validate(svd,surprise_data,measures=['RMSE', 'MAE'], cv=2, verbose=True)

"""> *RMSE measures the square root of the average squared differences between predicted and actual ratings. Lower values indicate better accuracy. The mean RMSE across the three folds is 0.7385, with a standard deviation of 0.0005, indicating very consistent performance across the folds.*

> *MAE measures the average absolute differences between predicted and actual ratings. Like RMSE, lower values indicate better accuracy. The mean MAE across the three folds is 0.5630, with a standard deviation of 0.0003, showing consistent performance.*

> *Fit time indicates the time taken to train the SVD model on each fold. The mean fit time is approximately 502.05 seconds, with some variability (standard deviation of 27.26 seconds) across the folds.*

> *Test time indicates the time taken to evaluate the SVD model on the test set of each fold. The mean test time is approximately 230.18 seconds, with more variability (standard deviation of 37.07 seconds) compared to the fit time.*
"""

# So first we take user 712664 and we try to recommend some movies based on the 5 star rated movies.

user_712664= merged_df[(merged_df['Cust_Id']==712664) & (merged_df['Rating']==5)]
user_712664=user_712664.set_index('Movie_Id')

user_712664

# Using the build_full_trainset method of the Surprise Dataset object (surprise_data)
# This method constructs a training set from the entire input data, suitable for training a model

train_set=surprise_data.build_full_trainset()

# Fitting the SVD algorithm (svd) to the training set (train_set)
svd.fit(train_set)

"""# Create Model that finds the best suited Movie for one  user in every genre."""

# Define a function to find the Top Movies for each User in every Genre.

def get_top_movies_per_genre(user_id, data ,algo):
    genres = data['Genres'].unique()
    top_movies = []

    # Get the set of movie IDs that the user has rated
    rated_movie_ids = set(data[data['Cust_Id'] == user_id]['Movie_Id'])
    avg_ratings = data.groupby('Movie_Id')['Rating'].mean().to_dict()

    for genre in genres:
        # Filter movies of the given genre
        genre_movie = data[data['Genres']==genre]['Movie_Id'].unique()

        # Using list comprehension to store only the movies that the user has not rated means not watched yet.
        unrated_genre_movies = [movie_id for movie_id in genre_movie if movie_id not in rated_movie_ids]

        # Predict ratings for the user for all movies in this genre.
        predictions = [algo.predict(user_id, movie_id) for movie_id in unrated_genre_movies]

        # Sort predictions by estimated score
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Get top movies for this genre if there are predictions available.
        if predictions:
            # Select the top movie prediction
            top_movie = predictions[0]

            # Get the title of the top movie
            movie_title= data[data['Movie_Id']==top_movie.iid]['Title'].iloc[0]

            avg_rating = avg_ratings.get(top_movie.iid, "No Rating")  # Get the average rating

            # Append top movie details to the list
            top_movies.append({'Movie_Id': top_movie.iid,
                               'Title': movie_title,
                               'Rating': avg_rating,
                               'Genre': genre,
                               'Estimated Score': top_movie.est})


    return pd.DataFrame(top_movies)

import random

# Assuming 'merged_df' is your DataFrame and it has a 'user_id' column
user_ids = merged_df['Cust_Id'].unique()
User_Id = random.choice(user_ids)

# The function returns a DataFrame containing the top movies per genre based on estimated scores from the SVD algorithm
top_movies_per_genre = get_top_movies_per_genre(User_Id, merged_df, svd)

# Sorting the DataFrame by 'Estimated Score' column in descending order
top_movies_per_genre = top_movies_per_genre.sort_values(by='Estimated Score', ascending=False).reset_index(drop=True)

# Displaying the DataFrame containing the top movies per genre
print(f"\033[1m Best Suited Movie For {User_Id} In Every Genre: ")
top_movies_per_genre.head(50)

"""# Checking if the user rated movies are present in the recommended movies.

"""

#Select the User
user=merged_df[merged_df['Cust_Id']==User_Id]

#Find the movies rated by user.
user_rated_movies=user['Movie_Id']

#Find the movies recommended to the user by the algorithm.
recommended_movies=top_movies_per_genre['Movie_Id']

#Check if the recommended movies are present in the user rated movies.
recommended_movies[recommended_movies.isin(user_rated_movies)