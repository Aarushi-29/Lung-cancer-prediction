# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import pickle
import numpy as np

loaded_model = pickle.load(open('/Users/pratyushkumargulzari/Downloads/depoly_lung_cancer_pred/trained_model.sav', 'rb'))

new_data = pd.DataFrame({
    'Age': [30],
    'Gender': [2],
    'Alcohol use': [6],
    'Genetic Risk': [4],
    'chronic Lung Disease': [3],
    'Wheezing': [2]
})

predictions = loaded_model.predict(new_data)

new_data['Predicted Level'] = predictions

predicted_messages = {
    "Low": "The person has a low chance of lung cancer.",
    "Medium": "The person has a moderate chance of lung cancer.",
    "High": "The person has a high chance of lung cancer."
}

new_data['Prediction Message'] = new_data['Predicted Level'].map(predicted_messages)
print(new_data)