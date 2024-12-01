# import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Promotion Prediction App")

# read the dataset to fill list values
df = pd.read_csv('train.csv')

# create input fields 
# categorical columns
Gender = st.selectbox("Gender", pd.unique(df['Gender']))
Vehicle_Age = st.selectbox("Vehicle_Age", pd.unique(df['Vehicle_Age']))
Vehicle_Damage = st.selectbox("Vehicle_Damage", pd.unique(df['Vehicle_Damage']))

# numerical columns
Age = st.number_input("Age")
Driving_License = st.number_input("Driving_License")
Region_Code = st.number_input("Region_Code")
Previously_Insured = st.number_input("Previously_Insured")
Annual_Premium = st.number_input("Annual_Premium")
Policy_Sales_Channel = st.number_input("Policy_Sales_Channel")
Vintage = st.number_input("Vintage")

# map the user inputs to respective columns
# convert the input values to dict
inputs = {
  "Gender": Gender,
  "Vehicle_Age": Vehicle_Age,
  "Vehicle_Damage": Vehicle_Damage,
  "Age": Age,
  "Driving_License": Driving_License,
  "Region_Code": Region_Code,
  "Previously_Insured": Previously_Insured,
  "Annual_Premium": Annual_Premium,
  "Policy_Sales_Channel": Policy_Sales_Channel,
  "Vintage": Vintage
}

# load the model from tje pickle file
model = joblib.load('janatahackcrosssell_pipeline_model.pkl')

# on click
if st.button("Predict"):
    # load the pickle model
    X_input = pd.DataFrame(inputs,index=[0])
    # predict the target using the loaded model
    prediction = model.predict(X_input)
    # display the result
    st.write("The predicted value is:")
    st.write(prediction)

# file upload to predict multiple values using test file
st.subheader("Please upload a csv file for prediction")
upload_file = st.file_uploader("Choose a csv file", type=['csv'])

if upload_file is not None:
    df = pd.read_csv(upload_file)

    st.write('File uploaded successfully')
    st.write(df.head(2))

    if st.button("Predict for the uploaded file"):
        df['Response'] = model.predict(df)
        st.write('Prediction completed')
        st.download_button(label="Download Prediction", data=df.to_csv(index=False),file_name="prediction.csv", mime="text/csv")
