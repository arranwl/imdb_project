import numpy as np
import pandas as pd
import streamlit as st
import own_functions as own

#Import data and create title and subheader
peopleData = pd.read_csv('clean_data/personScore.csv')

st.title('IMDb Prediction tool')
st.subheader('Welcome to the IMDb Prediction tool! Here you can input actors, actresses, '
           'and directors in an equation that returns the predicted IMDb score if it were a real movie.')

#Search and Select input
people = st.multiselect(label='What person(s) would you like?', options=peopleData['primaryName'], help='Type to search and add as many people as you would like!')

#When button is clicked it turns true and everything below happens
if st.button('Predict'):
    indices = own.get_list_indices(people)
    input = own.generate_input_data(indices)
    prediction = own.predict(input)
    #Ensure prediction printed isn't out of bounds
    if prediction > 10:
        prediction = 10
    elif prediction < 0:
        prediction = 0
    st.text("")
    st.subheader('Predicted IMDb Score: ' + str(round(prediction, 2)))
    coefs = own.get_coefs(people)
    st.text("")
    st.subheader('Coefficients of each peron:')
    for i in range(len(people)):
        st.text('-' + people[i] + ': ' + str(round(coefs[i], 3)))

#spacing
for i in range(15):
    st.text("")

#Information
with st.expander("About the project and creator"):
    st.write("This project was created from data made available by IMDb. The model used was a Linear"
             " Regression with Ridge Regularization. If you'd like to see the code that created the model"
             " and the UI, please click [here](https://github.com/arranwl/imdb_project)")
    st.write("About the creator:"
             " Arran Wass-Little is a third year student at the University of Florida undertaking a double"
             " major in Economics and Data Science. He's passionate about understanding the world through data"
             " whether through building tools or completing valuable research. If you'd like to contact him,"
             " please reach out to him at arranwasslittle@ufl.edu")