import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
import xgboost as xgb

# Function to get the absolute path of the project directory
def get_project_root():
    current_path = os.path.abspath(__file__)
    while True:
        current_path, tail = os.path.split(current_path)
        if 'models' in os.listdir(current_path):
            return current_path
        if tail == '':
            raise FileNotFoundError("Could not find the project directory with the 'models' folder")

# Adjust the path to access .pkl files
@st.cache_resource
def load_model():
    try:
        project_root = get_project_root()
        model_path = os.path.join(project_root, 'models', 'xgboost', 'model_xgboost.pkl')
        
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        raise

# Load the model
try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Could not load the model: {str(e)}")
    st.stop()

st.title('Stroke Risk Predictor')

st.write('Please enter the following patient data:')

# Mapeos para variables categóricas
gender_map = {'Male': 0, 'Female': 1}
ever_married_map = {'No': 0, 'Yes': 1}
work_type_map = {'Private': 3, 'Self-employed': 2, 'Govt_job': 0, 'children': 1}
residence_type_map = {'Urban': 1, 'Rural': 0}
smoking_status_map = {'never smoked': 3, 'formerly smoked': 1, 'smokes': 0, 'Unknown': 2}

# Create inputs for each feature
gender = st.selectbox('Gender', list(gender_map.keys()))
age = st.number_input('Age', min_value=0, max_value=120, value=30)
hypertension = st.selectbox('Hypertension', [0, 1])
heart_disease = st.selectbox('Heart disease', [0, 1])
ever_married = st.selectbox('Ever married', list(ever_married_map.keys()))
work_type = st.selectbox('Work type', list(work_type_map.keys()))
residence_type = st.selectbox('Residence type', list(residence_type_map.keys()))
avg_glucose_level = st.number_input('Average glucose level', min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
smoking_status = st.selectbox('Smoking status', list(smoking_status_map.keys()))

def validate_input(input_data):
    if input_data['age'].values[0] < 0 or input_data['age'].values[0] > 120:
        raise ValueError("Age must be between 0 and 120 years.")
    if input_data['avg_glucose_level'].values[0] < 0:
        raise ValueError("Glucose level cannot be negative.")
    if input_data['bmi'].values[0] < 10 or input_data['bmi'].values[0] > 50:
        raise ValueError("BMI must be between 10 and 50.")

# Store the threshold in the session state
if 'user_threshold' not in st.session_state:
    st.session_state.user_threshold = 0.5  # Default threshold

# Slider to adjust the dynamic threshold
st.session_state.user_threshold = st.slider('Adjust threshold for prediction', 0.0, 1.0, st.session_state.user_threshold)

if st.button('Predict'):
    # Create a DataFrame with the input data and mapped values
    input_data = pd.DataFrame({
        'gender': [gender_map[gender]],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married_map[ever_married]],
        'work_type': [work_type_map[work_type]],
        'Residence_type': [residence_type_map[residence_type]],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status_map[smoking_status]]
    })

    try:
        validate_input(input_data)
        
        st.write("Input data:")
        st.write(input_data)
        
        # Make the prediction
        probability = model.predict_proba(input_data)[0][1]
        
        # Use the threshold stored in session_state
        prediction = 1 if probability >= st.session_state.user_threshold else 0

        # Display results
        st.subheader('Prediction Results:')
        if prediction == 1:
            st.warning('The patient has a **high risk** of stroke.')
        else:
            st.success('The patient has a **low risk** of stroke.')

        st.write(f'**Estimated probability of stroke:** {probability:.2%}')
        st.write(f'**Selected decision threshold:** {st.session_state.user_threshold:.2%}')

        # Risk visualization based on probability
        st.write('Probability interpretation:')
        if probability < 0.3:
            st.write('**Low Risk**')
        elif 0.3 <= probability <= 0.7:
            st.write('**Moderate Risk**')
        else:
            st.write('**High Risk**')

        # Probability visualization
        fig, ax = plt.subplots()
        ax.bar(['No Stroke', 'Stroke'], [1-probability, probability])
        ax.set_ylabel('Probability')
        ax.set_title('Stroke Probability')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

st.info('Note: This tool is for informational purposes only and does not replace professional medical diagnosis.')