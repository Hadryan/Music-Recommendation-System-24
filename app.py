import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
load_dotenv()
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import xgboost

def song_to_df (sp, key):
    cat_cols = ['key', 'mode', 'time_signature']
    num_cols = ['danceability','energy','loudness','speechiness','acousticness',
                'instrumentalness','liveness','valence','tempo','duration_ms']

    row = pd.DataFrame(sp.audio_features(key)).drop(['type','uri',
                                               'track_href','analysis_url'], axis=1).set_index('id')
    return row

def make_genre_prediction(sp,key, ohe, model):
    cat_cols = ['key', 'mode', 'time_signature']
    num_cols = ['danceability','energy','loudness','speechiness','acousticness',
                'instrumentalness','liveness','valence','tempo','duration_ms']
    row = song_to_df(sp,key)
    temp_ohe = ohe.transform(row[cat_cols])
    returning_obj = row[num_cols].reset_index().join(pd.DataFrame(temp_ohe)).set_index('id')
    return model.predict(returning_obj)

def song_artist_from_key(sp,key):
    theTrack = sp.track(key)
    song_title = theTrack['name']
    artist_title = theTrack['artists'][0]['name']
    song_link = theTrack['external_urls']['spotify']
    return (song_title, artist_title, song_link)

def song_id_from_query(sp, query):
    q = query
    if(sp.search(q, limit=1, offset=0, type='track')['tracks']['total']>0):
        return sp.search( q, limit=1, offset=0, type='track')['tracks']['items'][0]['id']
    else:
        return None

def formatted_song_artist(sp, query):
    song_id = song_id_from_query(sp, query)
    song_artist = song_artist_from_key(sp, song_id)
    return (f"{song_artist[0]} by {song_artist[1]}")

scope = "user-library-read"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

infile = open('pickled_files/all_songs_genre_predicted.pickle','rb')
all_files = pickle.load(infile)
infile.close()

all_songs = all_files[0]
best_model = all_files[1]
ohe_make_genre_pred = all_files[2]

categorical_columns = list(all_songs.select_dtypes('object').columns)
numerical_columns = list(all_songs.select_dtypes(exclude = 'object').columns)

neigh = NearestNeighbors(n_neighbors=10, radius=0.45, metric='cosine')
X_knn = all_songs

MMScaler = preprocessing.MinMaxScaler()
MinMaxScaler = preprocessing.MinMaxScaler()
X_knn[numerical_columns] = MinMaxScaler.fit_transform(X_knn[numerical_columns])

ohe_knn = OneHotEncoder(drop='first', sparse=False)
X_knn_ohe = ohe_knn.fit_transform(X_knn[categorical_columns])
X_knn_transformed = X_knn[numerical_columns].reset_index().join(pd.DataFrame(X_knn_ohe, columns = ohe_knn.get_feature_names(categorical_columns))).set_index('id')

neigh.fit(X_knn_transformed)

def knn_preprocessing(sp, key, num_col = numerical_columns,
                      cat_col = categorical_columns,
                      mmScaler = MinMaxScaler, bm = best_model,
                      ohe_knn = ohe_knn, ohe_make_genre_pred = ohe_make_genre_pred):
    row = song_to_df(sp, key)
    genre = make_genre_prediction(sp,key, ohe_make_genre_pred, bm)
    st.write(f"Predicted Genre: {genre[0].title()}")
    row['predicted_genre'] = genre[0]
    row_dummied = ohe_knn.transform(row[cat_col])
    row[num_col] = mmScaler.transform(row[num_col])

    row = row[num_col].reset_index().join(pd.DataFrame(row_dummied, columns = ohe_knn.get_feature_names(cat_col))).set_index('id')
    return row

def make_song_recommendations(sp, kneighs, query):
    if(query.isspace() or not query):
        return "No results found"
    song_id = song_id_from_query(sp, query)
    if(song_id == None):
        return "No results found"
    song_plus_artist = song_artist_from_key(sp, song_id)
    song_to_rec = knn_preprocessing(sp, song_id)
    nbrs = neigh.kneighbors(
       song_to_rec, 15, return_distance=False
    )
    playlist = []
    for each in nbrs[0]:
        the_rec_song = song_artist_from_key(sp, X_knn_transformed.iloc[each].name)
        print((the_rec_song[0:2]))
        if (((the_rec_song[0:2]) != song_plus_artist[0:2]) and
           ((the_rec_song) not in playlist)):
            playlist.append(song_artist_from_key(sp, X_knn_transformed.iloc[each].name))
    return (playlist)


#Stream lit code

st.title("Music Recommendation System")

song_to_rec = st.text_input('Enter a song:')
if(song_to_rec):
    st.subheader(f"Songs recommended for {formatted_song_artist(sp,song_to_rec)} ")

    def make_clickable(url, text):
        return f'<a target="_blank" href="{url}">{text}</a>'

    # show data
    # st.write working links but no fixed height/width
    # st.spinner()
    # with st.spinner(text='Making Recommendations..'):
    #     msr = make_song_recommendations(sp, neigh, song_to_rec)
    #     msr = pd.DataFrame(msr, columns = ["Song", "Artist", "Link"])
    #     msr.index +=1
    #     msr['Link'] = msr['Link'].apply(make_clickable, args = ('Open Spotify',))
    #     st.write(msr.to_html(escape = False), unsafe_allow_html = True, width = 1000, height = 2000)

    #st.table no working links
    st.spinner()
    with st.spinner(text='Making Recommendations..'):
        msr = make_song_recommendations(sp, neigh, song_to_rec)
        msr = pd.DataFrame(msr, columns = ["Song", "Artist", "Link"])
        # msr['Link'] = msr['Link'].apply(lambda x: st.markdown(f'''[Open Spotify]({x})'''))
        # # msr['Link'] = msr['Link'].apply(lambda x: f'<a href="{x}">Open Spotify</a>')
        msr.index = msr.index+1

        st.table(msr)
    #     st.text(" \n")
