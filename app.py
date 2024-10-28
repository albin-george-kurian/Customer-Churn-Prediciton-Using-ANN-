import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import pickle

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('LabelEncoder_gender.pkl', 'rb') as file:
    label_encoder_gen = pickle.load(file)

with open('OneHotEncoder_geo.pkl', 'rb') as file:
    onehot_encoding_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)



# Creating streamlit app

st.title("End to End Customer churn prediction")

#user input
geography = st.selectbox('Geography', onehot_encoding_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gen.classes_)
age = st.slider('Age', 18,59)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1 , 4)
has_cr_card = st.selectbox('has credit card', [0, 1])
is_active_member = st.selectbox('Is active member', [0, 1])

# input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [label_encoder_gen.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# onehot encoding for geography
geo_encoder = onehot_encoding_geo.transform([[geography]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoder, columns=onehot_encoding_geo.get_feature_names_out(['Geography']))

# Combining onehot encoding for geography with user input
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoder_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn propability: {prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.write('The customer likly to churn')
else:
    st.write('The customer is not likely to churn')