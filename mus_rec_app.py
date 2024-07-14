import pandas as pd

df = pd.read_csv("spotify_millsongdata.csv")

# Data Cleaning
df = df.drop('link', axis=1).reset_index(drop=True)

# Sampling
df = df.sample(5000)

# Text Cleaning/Text Preprocessing
df['text'] = df['text'].str.lower().replace(r'^\w\s', ' ').replace(r'\n', ' ', regex=True)

# Tokenization
import nltk
from nltk.stem.porter import PorterStemmer

# Download the 'punkt' resource
nltk.download('punkt')  # Download the necessary resource for tokenization

stemmer = PorterStemmer()

def token(txt):
    token = nltk.word_tokenize(txt)
    a = [stemmer.stem(w) for w in token]
    return " ".join(a)

df['text'] = df['text'].apply(lambda x: token(x))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfid = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfid.fit_transform(df['text'])

# Cosine Similarity
similar = cosine_similarity(matrix)

# Recommendation
def recommendation(song_df):
    idx = df[df['song'] == song_df].index[0]
    distance = sorted(list(enumerate(similar[idx])), reverse=True, key=lambda x: x[1])
    songs = []
    for s_id in distance[1:5]:
        songs.append(df.iloc[s_id[0]].song)
    return songs

import pickle
pickle.dump(similar, open('similarity.pkl', 'wb'))
pickle.dump(df, open('df.pkl', 'wb'))

import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

CLIENT_ID = "e2dbf672f8ff47adb17049dac76c9af1"
CLIENT_SECRET = "48e2c76ef0a143e3bfeace7a2b0c511f"

# Initializing Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        print(album_cover_url)
        return album_cover_url
    else:
        return "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRslcO84eWfXP_4Ucd4Yfz6B8uqJmHaTo0iTw&s"

def recommend(song):
    index = df[df['song'] == song].index[0]
    distance = sorted(list(enumerate(similar[index])), reverse=True, key=lambda x: x[1])
    recommended_music_names = []
    recommended_music_posters = []
    for i in distance[1:6]:
        # Fetch the album cover
        artist = df.iloc[i[0]].artist
        print(artist)
        print(df.iloc[i[0]].song)
        recommended_music_posters.append(get_song_album_cover_url(df.iloc[i[0]].song, artist))
        recommended_music_names.append(df.iloc[i[0]].song)

    return recommended_music_names, recommended_music_posters

st.header('Music Recommender System')
music_list = df['song'].values
similarity = pickle.load(open('similarity.pkl', 'rb'))

selected_music = st.selectbox(
    "Type or select a song from the dropdown",
    music_list
)

if st.button('Show Recommendation'):
    recommended_music_names, recommended_music_posters = recommend(selected_music)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_music_names[0])
        st.image(recommended_music_posters[0])
    with col2:
        st.text(recommended_music_names[1])
        st.image(recommended_music_posters[1])
    with col3:
        st.text(recommended_music_names[2])
        st.image(recommended_music_posters[2])
    with col4:
        st.text(recommended_music_names[3])
        st.image(recommended_music_posters[3])
    with col5:
        st.text(recommended_music_names[4])
        st.image(recommended_music_posters[4])
