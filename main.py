import streamlit as st
from joblib import load
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

model = load(os.getcwd() + '/models/model.pkl')

st.title("Predict Risk of diabet")

age = st.slider("Age", 0, 100)
pregnancy = st.slider("Pregnancy", 0, 100)
glucose = st.number_input("Glucose")
insulin = st.number_input("Insulin")
blood = st.number_input("BloodPressure")
skin = st.number_input("SkinThickness")
bmi = st.number_input("BMI")
function = st.number_input("DiabetesPedigreeFunction")

if st.button("Predict"):
    if (age and pregnancy and glucose and insulin and blood and skin and bmi and function):
        input_data = {
            "age": [np.log(age)],
            "pregnancy" : [np.log(pregnancy)],
            "glucose" : [glucose],
            "insulin" : [np.log(insulin)],
            "blood" : [blood],
            "skin" : [skin],
            "bmi" : [bmi],
            "function" : [np.log(function)],
        }

        scaler = StandardScaler()
        X_std = scaler.fit_transform(input_data)

        input_df = pd.DataFrame(X_std)

        pred = model.predict(input_df)


        st.text(pred)


    else:
        st.text("Please select all values")