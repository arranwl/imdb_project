import numpy as np
import pandas as pd
import streamlit as st
import own_functions as own

peopleData = pd.read_csv('https://raw.githubusercontent.com/arranwl/imdb_project/main/clean_data/personScore.csv', index_col=0)

st.title('IMDb Prediction tool')
st.subheader('Welcome to the IMDb Prediction tool! Here you can input actors, actresses, '
           'and directors in an equation that returns the predicted IMDb score if it were a real movie.')

people = st.multiselect(label='What person would you like?', options=peopleData['primaryName'], help='Feel free to type to find your person!')

if st.button('Predict'):
    indices = own.get_list_indices(people)
    input = own.generate_input_data(indices)
    prediction = own.predict(input)
    st.write('Predicted IMDb Score:', str(round(prediction, 2)))


