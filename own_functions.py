import numpy as np
import pandas as pd
import pickle

filename = "model.pkl"
with open(filename, 'rb') as file:
    model = pickle.load(file)

peopleData = pd.read_csv('clean_data/personScore.csv')

def get_person_index(name):
    """Intakes a name and searches data for it's index value"""
    return peopleData.index[peopleData['primaryName'] == name].tolist()[0]

def get_list_indices(listNames):
    """Intakes a list of names and for each name applies get_person_index to generate a list of indices"""
    indices = [get_person_index(name) for name in listNames]
    return indices

def generate_input_data(indices):
    """Intakes a list of indices, and generates a numpy array with ones in the place of those indices
    such that the array is the sizee of input required for the model"""
    modelInput = np.zeros((1,10649))
    for index in indices:
        modelInput[0][index] = 1
    return modelInput

def predict(modelInput):
    """Intakes a numpy array of prepared data for prediction with the Model"""
    prediction = model.predict(modelInput)
    return prediction[0]

def get_coefs(listNames):
    """Intakes a list of names and searches through the data to find their coefficient in the model."""
    indices = get_list_indices(listNames)
    coefs = [model.coef_[index] for index in indices]
    return coefs

