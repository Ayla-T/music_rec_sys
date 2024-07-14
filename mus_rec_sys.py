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

stemmer = PorterStemmer()

def token(txt):
    token = nltk.word_tokenize(txt)
    a = [stemmer.stem(w) for w in token]
    return " ".join(a)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfid = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfid.fit_transform(df['text'])

