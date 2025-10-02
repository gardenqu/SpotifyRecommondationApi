# -*- coding: utf-8 -*-
"""KNN.ipynb - Flask API for music recommendation based on cosine similarity"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load and preprocess dataset
dataFrame = pd.read_csv('spotify_scaled_df.csv')
scaler = joblib.load('scaler.pkl')

# Feature categories

numeric_features = ['acousticness', 'danceability', 'energy',
                    'instrumentalness', 'key', 'liveness', 'loudness',
                    'mode', 'speechiness', 'tempo', 'valence']

# Normalize numeric features
scaler = StandardScaler()
dataFrame[numeric_features] = scaler.fit_transform(dataFrame[numeric_features])

dataFrame['track_name'] = dataFrame['track_name'].str.lower()
dataFrame['artist_name'] = dataFrame['artist_name'].str.lower()

# Get song vector for similarity computation
def get_song_vector(track_name, artist_name, df):
    # Normalize input
    track_name = track_name.strip().lower()
    artist_name = artist_name.strip().lower()

    # Normalize DataFrame columns
    df['track_name'] = df['track_name'].str.strip().str.lower()
    df['artist_name'] = df['artist_name'].str.strip().str.lower()

    # Filter rows using substring match
    song = df[
        df['track_name'].str.contains(track_name, na=False) &
        df['artist_name'].str.contains(artist_name, na=False)
    ]

    if song.empty:
        return None

    return song[numeric_features].values


# Recommend similar songs based on cosine similarity
def recommend_similar_songs(track_name, artist_name, df, top_n=5):

    query_vector = get_song_vector(track_name, artist_name, df)
    if query_vector is None:
        return None

    candidate_vectors = df[numeric_features].values
    similarity_scores = cosine_similarity(query_vector, candidate_vectors)[0]

    df['similarity_score'] = similarity_scores
    recommendations = df[df['track_name'] != track_name]  # Exclude the query song
    return recommendations.sort_values(by='similarity_score', ascending=False).head(top_n)[
        ['track_name', 'artist_name', 'similarity_score', 'track_url']
    ]



# Flask route for recommendation API
@app.route('/recommend_similar', methods=['POST'])
def recommend_similar():
    data = request.get_json()

    track_name = data.get('track_name')
    artist_name = data.get('artist_name')
    top_n = data.get('top_n', 5)

    if not track_name or not artist_name:
        return jsonify({'error': 'track_name and artist_name are required'}), 400

    recommendations_df = recommend_similar_songs(track_name, artist_name, dataFrame, top_n)
    
    if recommendations_df is None:
        return jsonify({'error': 'Song not found.'}), 404

    recommendations = recommendations_df.to_dict(orient='records')
    return jsonify({'recommendations': recommendations})



@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    track_name = data['track_name']
    artist_name = data['artist_name']

    
   

    query_vector = get_song_vector(track_name, artist_name, dataFrame)
    if query_vector is None:
        return jsonify({'error': 'Song not found'}), 404

    candidate_vectors = dataFrame[numeric_features].values
    similarity_scores = cosine_similarity(query_vector, candidate_vectors)[0]

    print(type(similarity_scores)) 

    
    dataFrame['similarity_score'] = similarity_scores
    recommendations = dataFrame[dataFrame['track_name'] != track_name]
    results = recommendations.sort_values(by='similarity_score', ascending=False).head(5)[
        ['track_name', 'artist_name', 'similarity_score', 'track_url', 'artwork_url']
    ]
    return jsonify(results.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)