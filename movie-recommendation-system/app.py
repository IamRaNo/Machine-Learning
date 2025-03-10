import streamlit as st
import pickle

st.title('Movie Recommended System')
data = pickle.load(open('data.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

movies_list = data['title'].values
selected_movie = st.selectbox('Enter movie name', movies_list)


def recommend(movie):
    recommendation_list = []
    movie_index = data[data['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for movie in movie_list:
        recommendation_list.append(data.iloc[movie[0]].title)
    return recommendation_list


if st.button('Recommend'):
    recommendation = recommend(selected_movie)
    for i in recommendation:
        st.write(i)
