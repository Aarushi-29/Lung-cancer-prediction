#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 02:13:20 2023

@author: pratyushkumargulzari
"""
import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('/Users/pratyushkumargulzari/Downloads/depoly_lung_cancer_pred/trained_model.sav', 'rb'))

def lung_cancer_prediction(new_data):
    new_data = np.array(new_data).reshape(1, -1)
    predictions = loaded_model.predict(new_data)

    predicted_messages = {
        "Low": "The person has a low chance of lung cancer.",
        "Medium": "The person has a moderate chance of lung cancer.",
        "High": "The person has a high chance of lung cancer."
    }

    prediction_message = predicted_messages.get(predictions[0], "Invalid prediction")
    return prediction_message
    
def main():
    st.title('Lung Cancer Prediction')
    
    # Create number input fields for various features
    age = st.number_input('Age', help='Enter the age in years')
    gender = st.number_input('Gender', help='Enter 1 for Male and 2 for Female')
    alcohol_use = st.number_input('Alcohol use', help='Enter the level of alcohol use (0 to 10)')
    genetic_risk = st.number_input('Genetic Risk', help='Enter the genetic risk level (0 to 10)')
    chronic_lung_disease = st.number_input('Chronic Lung Disease', help='Enter the severity of chronic lung disease (0 to 10)')
    wheezing = st.number_input('Wheezing', help='Enter the wheezing level (0 to 10)')
    
    diagnosis = ''
    
    if st.button('Lung Cancer Prediction'):
        diagnosis = lung_cancer_prediction([age, gender, alcohol_use, genetic_risk, chronic_lung_disease, wheezing])
        
        st.success(diagnosis)
            
if __name__ == '__main__':
    main()



