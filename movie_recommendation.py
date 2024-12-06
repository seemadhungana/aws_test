import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer

# load the datasets
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")

# process movies_df: extract year, title, and split genres
movies_df["year"] = movies_df["title"].str.extract(r'\((\d{4})\)') 
movies_df["title"] = movies_df["title"].str.rsplit(" (", n=1).str[0]
movies_df["genres"] = movies_df["genres"].str.split("|")

# compute average rating for each movie
average_ratings_df = ratings_df.groupby("movieId", as_index=False)["rating"].mean()
average_ratings_df.rename(columns={"rating": "average_rating"}, inplace=True)

# merge movies and average ratings into a summary DataFrame
summary_df = pd.merge(movies_df, average_ratings_df, on="movieId", how="inner")

# feature engineering: convert genres into a single string for vectorization
summary_df["genres_str"] = summary_df["genres"].apply(lambda x: " ".join(x))

# combine genres and average rating into a feature for KNN
cv = CountVectorizer()
genre_features = cv.fit_transform(summary_df["genres_str"])

# combine genre vectors with average ratings for similarity
features = np.hstack([genre_features.toarray(), summary_df[["average_rating"]].values])

# train KNN model
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(features)

def recommend_movies(movie_title, summary_df, knn_model, feature_matrix):
    try:
        # Find the index of the input movie
        movie_index = summary_df[summary_df["title"].str.contains(movie_title, case=False, na=False)].index[0]
        distances, indices = knn_model.kneighbors([feature_matrix[movie_index]])
        
        # Get the recommended movie indices
        recommended_indices = indices[0][1:]  # Exclude the input movie
        recommendations = summary_df.iloc[recommended_indices]
        return recommendations
    except IndexError:
        return None

st.title("Movie Recommendation System")
st.write("Find similar movies based on genres and average ratings.")

movie_input = st.text_input("Enter a movie you like:")

if movie_input:
    recommendations = recommend_movies(movie_input, summary_df, knn, features)
    
    if recommendations is not None and not recommendations.empty:
        st.write("### Recommendations:")
        for _, row in recommendations.iterrows():
            st.write(f"**{row['title']} ({row['year']})** - Genres: {', '.join(row['genres'])}")
    else:
        st.write("Movie not found. Please try another title.")
