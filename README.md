# 🎵 Music Recommendation API

A Flask-based REST API that recommends similar songs using **cosine similarity** on Spotify audio features.  
It takes a track name and artist name as input and returns the top 5 most similar songs.

---

## 🚀 Features
- ✅ Get list of all available songs  
- ✅ Get list of all available artists  
- ✅ Recommend similar tracks given a `trackName` and `trackArtist`  
- ✅ Returns track metadata:  
  - `track_name`  
  - `artist_name`  
  - `similarity_score`  
  - `track_url`  
  - `artwork_url`  

---

## 📂 Project Structure
.
├── KNN.ipynb # Flask API implementation
├── spotify_scaled_df.csv # Preprocessed dataset with Spotify tracks
├── scaler.pkl # Saved StandardScaler for feature normalization
