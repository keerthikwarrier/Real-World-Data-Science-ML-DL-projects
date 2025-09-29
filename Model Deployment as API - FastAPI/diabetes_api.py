# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 11:37:53 2025

@author: Keert
"""

from fastapi import FastAPI   # FastAPI framework for building APIs (routes, endpoints, etc.)
from pydantic import BaseModel  # Used to define data models with validation (request/response schemas)
import pickle   # For serializing and deserializing Python objects (e.g., saving/loading ML models)
import json     # For working with JSON data (encode/decode request/response payloads)


app = FastAPI()

""" Mentioning the input format needed for the model"""
class model_input(BaseModel):
    
    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int       
        
# loading the saved model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# The input data will be sent in the model_input format to post api and /diabetesModel_prediction will be the endpoint of the api

@app.post('/diabetesModel_prediction')
def diabetes_pred(inputParameters : model_input): #The inputparameters are the place where the new input data sent by the user in post method is saved"""
    
    #The data will be posted to API in json format"""
    input_data=inputParameters.json()
    input_dictionary=json.loads(input_data)
    
    """preg = input_dictionary['pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']
    
    
    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]"""
    
    """input_list = [input_dictionary[key] for key in input_dictionary.items()]"""
    
    input_list = [input_dictionary[key] for key in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    
    prediction=diabetes_model.predict([input_list])
    if(prediction[0]==0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"

