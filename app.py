import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model (update the file path as necessary)
model_path = 'heart_disease_model3.joblib'
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Define input fields for the user to enter data
st.title('Heart Disease Prediction')

st.write("Please provide the following details:")

user_name = st.text_input('Name')
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
smoking = st.selectbox('Smoking', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
alcohol_drinking = st.selectbox('Alcohol Drinking', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
stroke = st.selectbox('Stroke', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
physical_health = st.number_input('Physical Health (Number of days with bad physical health in the past 30 days)', min_value=0, max_value=30, value=0)
mental_health = st.number_input('Mental Health (Number of days with bad mental health in the past 30 days)', min_value=0, max_value=30, value=0)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
age_category = st.selectbox('Age Category', options=[
    '1-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'
], format_func=lambda x: x)
diabetic = st.selectbox('Diabetic', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
physical_activity = st.selectbox('Physical Activity', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
asthma = st.selectbox('Asthma', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
kidney_disease = st.selectbox('Kidney Disease', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Map age ranges to categories
age_category_mapping = {
    '1-10': 1,
    '10-20': 2,
    '20-30': 3,
    '30-40': 4,
    '40-50': 5,
    '50-60': 6,
    '60-70': 7,
    '70-80': 8,
    '80-90': 9,
    '90-100': 10
}
age_category_value = age_category_mapping[age_category]

# Collect user input into a dataframe
user_data = pd.DataFrame({
    'BMI': [bmi],
    'Smoking': [smoking],
    'AlcoholDrinking': [alcohol_drinking],
    'Stroke': [stroke],
    'PhysicalHealth': [physical_health],
    'MentalHealth': [mental_health],
    'Sex': [sex],
    'AgeCategory': [age_category_value],
    'Diabetic': [diabetic],
    'PhysicalActivity': [physical_activity],
    'Asthma': [asthma],
    'KidneyDisease': [kidney_disease]
})

# Ensure the order of columns matches the model's expected input
user_data = user_data[['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'Sex', 'AgeCategory', 'Diabetic', 'PhysicalActivity', 'Asthma', 'KidneyDisease']]

# Display user input
st.subheader('User Input:')
st.write(f"**Name:** {user_name}")
st.write(user_data)

# Convert user_data to numpy array and reshape
input_data_as_numpy_array = user_data.values
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Debug: Print input data to verify
st.write("Input Data for Prediction:")
st.write(input_data_reshaped)

# Predict heart disease
if st.button('Predict'):
    if model is not None:
        try:
            prediction = model.predict(input_data_reshaped)[0]
            prediction_proba = model.predict_proba(input_data_reshaped)[0]

            st.subheader('Prediction:')
            st.write('Heart Disease' if prediction == 1 else 'No Heart Disease')

            st.subheader('Prediction Probability:')
            st.write(f'Probability of Heart Disease: {prediction_proba[1]:.2f}')
            st.write(f'Probability of No Heart Disease: {prediction_proba[0]:.2f}')

            # Display a bar graph of the input parameters
            st.subheader('User Input Parameters:')
            fig, ax = plt.subplots(figsize=(10, 5))
            user_data.T.plot(kind='bar', ax=ax, color=[
                'skyblue', 'salmon', 'lightgreen', 'gold', 'violet', 
                'lightcoral', 'deepskyblue', 'lightpink', 'lightyellow', 
                'lightgray', 'lightblue', 'lightsteelblue'
            ])
            ax.set_title(f'Input Parameters for {user_name}')
            ax.set_xlabel('Parameters')
            ax.set_ylabel('Values')
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Model is not loaded. Please check the model file and try again.")
