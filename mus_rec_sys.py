import pandas as pd

df = pd.read_csv("spotify_millsongdata.csv")

#Data Cleaning
df = df.drop('link', axis = 1).reset_index(drop = True)

#Sampling
df = df.sample(5000)

#Text Cleaning/Text Preprocessing
df['text'] = df['text'].str.lower().replace(r'^\w\s',' ').replace(r'\n', ' ', regex= True)

#Tokenization
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

#Cosine Similarity
similar = cosine_similarity(matrix)

#Recommendation Function
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

