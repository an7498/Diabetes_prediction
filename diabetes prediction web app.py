# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:58:56 2025

@author: Anmol
"""

import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open('F:/OneDrive/Desktop/ml project/finalized_model.sav', 'rb'))
#creating a function
def diabetes_prediction(input_data):
    input_data=(10,168,74,0,0,38,0.537,34)
    #change input data as numpy array
    input_data_as_numpy_array=np.asarray(input_data)
    #reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    #standarize the input data


    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0]==0):
      return ' According to ML prediction,the person is not diabetic'
    else:
      return 'According to ML prediction,the person is diabetic'

def main():
    
    #giving a title
    st.title('Diabetes Prediction web application')
    #creating the input data from the user
    
    Pregnancies=st.text_input("Number of pregnancies")
    Glucose=st.text_input("Glucose level")
    BloodPressure=st.text_input("BP level")
    SkinThickness=st.text_input("Skin Thickness")
    Insulin=st.text_input("Insulin level")
    BMI=st.text_input("BMI index") 
    DiabetesPedigreeFunction=st.text_input("Diabetes Pedigree Function value")
    Age=st.text_input("Age")
    #code for prediction
    diagnosis=''
    #creating a button fro prediction
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction( [Pregnancies,Glucose,BloodPressure,SkinThickness, Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
    

        
        