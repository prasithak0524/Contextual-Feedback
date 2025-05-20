# recommender.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    df = pd.read_csv("data/imdb_2024_movies.csv")
    df.dropna(subset=["Storyline"], inplace=True)
    return df

def get_recommendations(user_input, df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["Storyline"])

    user_vec = tfidf.transform([user_input])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()

    top_indices = cosine_sim.argsort()[-5:][::-1]
    return df.iloc[top_indices][["Movie Name", "Storyline"]]
