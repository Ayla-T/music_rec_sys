import pandas as pd
df = pd.read_csv("spotify_millsongdata.csv")
df = df.drop('link', axis = 1).reset_index(drop = True)
df = df.sample(5000)
df['text'][0]
#Text Cleaning/Text Preprocessing
df['text'] = df['text'].str.lower().replace(r'^\w\s',' ').replace(r'\n', ' ', regex= True)