#Import the necessary libraries
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

# Convert a song_id to a dataframe row
def song_to_df (sp, key):
    cat_cols = ['key', 'mode', 'time_signature']
    num_cols = ['danceability','energy','loudness','speechiness','acousticness',
                'instrumentalness','liveness','valence','tempo','duration_ms']

    row = pd.DataFrame(sp.audio_features(key)).drop(['type','uri',
                                               'track_href','analysis_url'], axis=1).set_index('id')
    return row

# Do preprocessing and make a genre prediction for a song
def make_genre_prediction(sp,key, ohe, model):
    cat_cols = ['key', 'mode', 'time_signature']
    num_cols = ['danceability','energy','loudness','speechiness','acousticness',
                'instrumentalness','liveness','valence','tempo','duration_ms']
    row = song_to_df(sp,key)
    temp_ohe = ohe.transform(row[cat_cols])
    returning_obj = row[num_cols].reset_index().join(pd.DataFrame(temp_ohe)).set_index('id')
    return model.predict(returning_obj)

# Get the song info from song_id
def song_artist_from_key(sp,key):
    theTrack = sp.track(key)
    if(theTrack is not None):
        song_title = theTrack['name']
        artist_title = theTrack['artists'][0]['name']
        song_link = theTrack['external_urls']['spotify']
        return (song_title, artist_title, song_link)
    else:
        return None

# Get the song id from a query
def song_id_from_query(sp, query):
    q = query
    if(sp.search(q, limit=1, offset=0, type='track')['tracks']['total']>0):
        return sp.search( q, limit=1, offset=0, type='track')['tracks']['items'][0]['id']
    else:
        return None

# Return formatted song and artist as string
def formatted_song_artist(sp, query):
    song_id = song_id_from_query(sp, query)
    if(song_id):
        song_artist = song_artist_from_key(sp, song_id)
        return (f"{song_artist[0]} by {song_artist[1]}")
    else:
        return None
# Authorize spotify api object
scope = "user-library-read"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

# Import necessary pickled files
infile = open('pickled_files/all_songs_genre_predicted.pickle','rb')
all_files = pickle.load(infile)
infile.close()

all_songs = all_files[0]
best_model = all_files[1]
ohe_make_genre_pred = all_files[2]

# Create variables to easily access categorical and numerical columns
categorical_columns = list(all_songs.select_dtypes('object').columns)
numerical_columns = list(all_songs.select_dtypes(exclude = 'object').columns)

# Create a nearest neighbors object using cosine similarity metric.
neigh = NearestNeighbors(n_neighbors=10, radius=0.45, metric='cosine')
X_knn = all_songs

# Total dataframe normalizing
MMScaler = preprocessing.MinMaxScaler()
MinMaxScaler = preprocessing.MinMaxScaler()
X_knn[numerical_columns] = MinMaxScaler.fit_transform(X_knn[numerical_columns])

# Total dataframe dummying
ohe_knn = OneHotEncoder(drop='first', sparse=False)
X_knn_ohe = ohe_knn.fit_transform(X_knn[categorical_columns])
X_knn_transformed = X_knn[numerical_columns].reset_index().join(pd.DataFrame(X_knn_ohe, columns = ohe_knn.get_feature_names(categorical_columns))).set_index('id')

# Fit the model
neigh.fit(X_knn_transformed)

# Do proper preprocessing for a single song
def knn_preprocessing(sp, key, num_col = numerical_columns,
                      cat_col = categorical_columns,
                      mmScaler = MinMaxScaler, bm = best_model,
                      ohe_knn = ohe_knn, ohe_make_genre_pred = ohe_make_genre_pred):
    # Convert song to the dataframe
    row = song_to_df(sp, key)
    # Make genre prediction for inputted song
    genre = make_genre_prediction(sp,key, ohe_make_genre_pred, bm)
    # Write predicted genre to page
    st.write(f"Predicted Genre: {genre[0].title()}")
    # Append the predicted genre
    row['predicted_genre'] = genre[0]
    # Dummy the categorical
    row_dummied = ohe_knn.transform(row[cat_col])
    # Normalize the numerical
    row[num_col] = mmScaler.transform(row[num_col])
    # Combine the preprocessed rows and return it
    row = row[num_col].reset_index().join(pd.DataFrame(row_dummied, columns = ohe_knn.get_feature_names(cat_col))).set_index('id')
    return row

def make_song_recommendations(sp, kneighs, query):
    #If the query is aspace or not filled, return no results
    if(query.isspace() or not query):
        return "No results found"
    song_id = song_id_from_query(sp, query)
    # If the query doesn't return an id, return no results
    if(song_id == None):
        return "No results found"
    # Get the song info
    song_plus_artist = song_artist_from_key(sp, song_id)
    # Preprocess the tracks
    song_to_rec = knn_preprocessing(sp, song_id)
    # Get the 15 nearest neighbors to inputted song
    nbrs = neigh.kneighbors(
       song_to_rec, 15, return_distance=False
    )
    # Properly retrieve the song info of each neighbor and return it
    playlist = []
    for each in nbrs[0]:
        the_rec_song = song_artist_from_key(sp, X_knn_transformed.iloc[each].name)
        print((the_rec_song[0:2]))
        if (((the_rec_song[0:2]) != song_plus_artist[0:2]) and
           ((the_rec_song) not in playlist)):
            playlist.append(song_artist_from_key(sp, X_knn_transformed.iloc[each].name))
    return (playlist)


#Stream lit code

# Set title for page
st.title("Music Recommendation System")

# Ask user to input song
song_to_rec = st.text_input('Enter a song:')

# Only run code if there was text inputted
if(song_to_rec):
    st.subheader(f"Songs recommended for {formatted_song_artist(sp,song_to_rec)} ")

    # Commented out code refers to a different version of the table printed
    # .. to the page with working links but not great style.

    # def make_clickable(url, text):
    #     return f'<a target="_blank" href="{url}">{text}</a>'

    # show data to page

    # st.write working links but no fixed height/width
    # st.spinner()
    # with st.spinner(text='Making Recommendations..'):
    #     msr = make_song_recommendations(sp, neigh, song_to_rec)
    #     msr = pd.DataFrame(msr, columns = ["Song", "Artist", "Link"])
    #     msr.index +=1
    #     msr['Link'] = msr['Link'].apply(make_clickable, args = ('Open Spotify',))
    #     st.write(msr.to_html(escape = False), unsafe_allow_html = True, width = 1000, height = 2000)

    # show data to page
    #st.table no working links
    st.spinner()
    with st.spinner(text='Making Recommendations..'):
        msr = make_song_recommendations(sp, neigh, song_to_rec)

        if(msr != "No results found"):
            msr = pd.DataFrame(msr, columns = ["Song", "Artist", "Link"])
            msr.index = msr.index+1
            st.table(msr)
        else:
            st.write(msr)
