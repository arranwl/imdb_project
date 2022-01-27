import numpy as np
import pandas as pd
import streamlit as st
import own_functions as own



peopleData = pd.read_csv('clean_data/personScore.csv')

st.title('IMDb Prediction tool')
st.subheader('Welcome to the IMDb Prediction tool! Here you can input actors, actresses, '
           'and directors in an equation that returns the predicted IMDb score if it were a real movie.')

people = st.multiselect(label='What person would you like?', options=peopleData['primaryName'])

if st.button('Predict'):
    indices = own.get_list_indices(people)
    input = own.generate_input_data(indices)
    prediction = own.predict(input)
    if prediction>10:
        prediction = 10
    st.text("")
    test = str(round(prediction, 2))
    st.subheader('Predicted IMDb Score: ' + test)
    coefs = own.get_coefs(people)
    st.text("")
    st.subheader('Coefficients of each peron:')
    for i in range(len(people)):
        st.text('-' + people[i] + ': ' + str(round(coefs[i], 3)))


