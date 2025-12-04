import streamlit as st
from joblib import load
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

model = load(os.getcwd() + '/models/model.pkl')
scaler = load(os.getcwd() + '/models/scaler.pkl')

st.title("Prédire le risque de diabète")

age = st.slider("Age", 0, 100)
pregnancy = st.slider("Pregnancy", 0, 100)
glucose = st.number_input("Glucose")
insulin = st.number_input("Insulin")
blood = st.number_input("BloodPressure")
skin = st.number_input("SkinThickness")
bmi = st.number_input("BMI")
function = st.number_input("DiabetesPedigreeFunction")

if st.button("Prédire"):
    if (age and pregnancy and glucose and insulin and blood and skin and bmi and function):
        input_data = {
            "Pregnancies" : [np.log(pregnancy)],
            "Glucose" : [glucose],
            "BloodPressure" : [blood],
            "SkinThickness" : [skin],
            "Insulin" : [np.log(insulin)],
            "BMI" : [bmi],
            "DiabetesPedigreeFunction" : [np.log(function)],
            "Age": [np.log(age)],
        }

        input_df = pd.DataFrame(input_data)

        X_std = scaler.transform(input_df)

        pred = model.predict(X_std)

        st.divider()
        st.subheader("Le résultat")

        if pred[0] == 1:
            st.error("Le risque d'avoir le diabete est: **Eleve**")
        else:
            st.success("Le risque d'avoir le diabete est: **Faible**")


    else:
        st.text("Please select all values")