import numpy as np
import pandas as pd
import pickle

filename = "model.pkl"
with open(filename, 'rb') as file:
    model = pickle.load(file)

peopleData = pd.read_csv('clean_data/personScore.csv')

def get_person_index(name):
    return peopleData.index[peopleData['primaryName'] == name].tolist()[0]

def get_list_indices(listNames):
    indices = [get_person_index(name) for name in listNames]
    return indices

def generate_input_data(indices):
    modelInput = np.zeros((1,10649))
    for index in indices:
        modelInput[0][index] = 1
    return modelInput

def predict(modelInput):
    prediction = model.predict(modelInput)
    return prediction[0]

