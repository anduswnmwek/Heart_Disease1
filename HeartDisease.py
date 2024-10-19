import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st


logo = Image.open('image (6).png')
resized_logo = logo.resize((150, 105))#447 × 559
st.sidebar.image(resized_logo, use_column_width=False)
# Title and description
st.title("Heart Disease Prediction App")
st.write("This app predicts heart disease likelihood based on input features from preprocessed data.")
st.markdown("**Done by Juhan , Ryan ,Shaurya ,kaushiki**" )
# Load the trained model
model_path = 'heart_disease_model1.pkl'
logreg = joblib.load(model_path)

# Sidebar for user input features
st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.slider('Age (Normalized)', 0.0, 1.0, 0.5)
    sex = st.sidebar.selectbox('Sex_2 (0: Female, 1: Male)', [0.0, 1.0])
    if sex == 'Male':
        sex_1 = 1.0
        sex_2 = 0.0
    else:
        sex_1 = 0.0
        sex_2 = 1.0
    chest_pain = st.sidebar.slider('Chest Pain Type (0-1 normalized)', 0.0, 1.0, 0.5)
    fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (0: No, 1: Yes)', [0.0, 1.0])
    resting_ecg = st.sidebar.selectbox('Resting Electrocardiographic Results (0-1 normalized)', [0.0, 1.0])
    max_hr = st.sidebar.slider('Max Heart Rate Achieved (Normalized)', 0.0, 1.0, 0.5)
    exercise_angina = st.sidebar.selectbox('Exercise Angina_1 (0: No, 1: Yes)', [0.0, 1.0])
    if exercise_angina == 'Yes':
        exercise_angina_1 = 1.0
        exercise_angina_2 = 0.0
    else:
        exercise_angina_1 = 0.0
        exercise_angina_2 = 1.0
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise (Normalized)', 0.0, 1.0, 0.5)
    st_slope = st.sidebar.slider('ST Slope (0-1 normalized)', 0.0, 1.0, 0.5)
    diastolic_bp = st.sidebar.slider('Diastolic Resting BP (Normalized)', 0.0, 1.0, 0.5)
    resting_bp_cat = st.sidebar.slider('Resting BP Category (Normalized)', 0.0, 1.0, 0.5)
    chol_cat = st.sidebar.slider('Cholesterol Category (Normalized)', 0.0, 1.0, 0.5)
    
    data = {
        'Age': age,
        'Sex_1': sex_1,
        'Sex_2': sex_2,
        'ChestPainType': chest_pain,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina_1': exercise_angina_1,
        'ExerciseAngina_2': exercise_angina_2,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope,
        'Diastolic_RestingBP': diastolic_bp,
        'RestingBP_Category': resting_bp_cat,
        'Cholesterol_Category': chol_cat
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user inputs in the main area
st.subheader('User Input Features')
st.write(input_df)

# Make predictions using the loaded model
prediction = logreg.predict(input_df)
prediction_proba = logreg.predict_proba(input_df)

st.subheader('Prediction')
if prediction == 1:
    st.write("The model predicts that you may have heart disease.")
else:
    st.write("The model predicts that you are not likely to have heart disease.")

# Display the prediction probability
st.write(f"Prediction probability: {prediction_proba[0][1] * 100:.2f}% likelihood of heart disease")

image='doc.png'
st.write('------------------------------------------------------------------------------------------------------------------------------------------------------------------')
st.subheader("**Reviews:** ")

image = Image.open(image)
st.image(image)

