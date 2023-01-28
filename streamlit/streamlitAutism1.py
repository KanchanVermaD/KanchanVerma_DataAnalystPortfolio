# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import joblib
def predict(data):
    clf = joblib.load('D:/BITS MTech/Sem4/Dataset/trained_file3.sav', 'rb')
    return clf.predict([a,b,c,d,e,f,g,h,i,j])
#loaded_model2=pickle.load(open('D:/BITS MTech/Sem4/Dataset/trained_file3.sav', 'rb'))
 
page = st.sidebar.selectbox("""
         Hello there! I’ll guide you!
         Please select model""", 
         ["Main Page", 
          "Model dashboard"])
header= st.container()
dataset= st.container()
features=st.container()
model_training=st.container()
with header:
    st.title(" Autism Prediction- Using a machine learning approach")
    st.image("https://blogs.biomedcentral.com/on-biology/wp-content/uploads/sites/5/2018/03/feature-image-square-1-620x342.jpg" )

    
      
     
# with model_training:
    # changing the input_data to numpy array
    # def autism_prediction(input_data):
    #     input_data_as_numpy_array = np.asarray(input_data)

    #     # reshape the array as we are predicting for one instance
    #     input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    #     prediction = loaded_model2.predict(input_data_reshaped)
    #     print(prediction)
    #     if (prediction[0] == 0):
    #         print('The person is not autistic')
    #     else:
    #         print('The person is autistic')

def main():
    st.title("Autism prediction web app")
    A1_Score = st.multiselect('It is difficult for me to understand how other people are feeling when we are talking: ',
              ('True now and when I was young', 'True only now', 'True when I was <16', 'Never true'))
 
    A2_Score = st.multiselect('Some ordinary textures that do not bother others feel very offensive to me: ',
              ('True now and when I was young', 'True only now', 'True when I was <16', 'Never true'))

    A3_Score = st.multiselect('It is difficult for me to work and function in groups: ',
              ('True now and when I was young', 'True only now', 'True when I was <16', 'Never true'))
    A4_Score = st.multiselect('It is difficult to figure out what other people expect of me: ',
              ('True now and when I was young', 'True only now', 'True when I was <16', 'Never true'))
    A5_Score = st.multiselect('I often do not know how to act in social situations: ',
              ('True now and when I was young', 'True only now', 'True when I was <16', 'Never true'))
    A6_Score = st.multiselect('I can chat and make small talk with people: ',
              ('True now and when I was young', 'True only now', 'True when I was <16', 'Never true'))
    A7_Score = st.multiselect('When I feel overwhelmed by my senses, I have to isolate myself to shut them down: ',
              ('True now and when I was young', 'True only now', 'True when I was <16', 'Never true'))
    A8_Score = st.multiselect('How to make friends and socialize is a mystery for me: ',
              ('True now and when I was young', 'True only now', 'True when I was <16', 'Never true'))
    A9_Score = st.multiselect('When talking to someone, I have a hard time telling when it is my turn to talk or to listen: ',
              ('True now and when I was young', 'True only now', 'True when I was <16', 'Never true'))
    A10_Score = st.multiselect('Sometimes I have to cover my ears to block out painful noises(like vaccum cleaners or people talking too much or too loudly): ',
              ('True now and when I was young', 'True only now', 'True when I was <16', 'Never true'))
    if A1_Score == "True now and when I was young":
        A1_Score = 0
    elif A1_Score == 'True only now':
        A1_Score = 0
    elif A1_Score == 'True when I was <16':
        A1_Score = 1
    else:
        A1_Score = 1
   
    if A2_Score == "True now and when I was young":
        A2_Score = 0
    elif A2_Score == 'True only now':
        A2_Score = 0
    elif A2_Score == 'True when I was <16':
        A2_Score = 1
    else:
        A2_Score = 1
     
    if A3_Score == "True now and when I was young":
        A3_Score = 0
    elif A3_Score == 'True only now':
        A3_Score = 0
    elif A3_Score == 'True when I was <16':
        A3_Score = 1
    else:
        A3_Score = 1
    
    if A4_Score == "True now and when I was young":
        A4_Score = 0
    elif A4_Score == 'True only now':
        A4_Score = 0
    elif A4_Score == 'True when I was <16':
        A4_Score = 1
    else:
        A4_Score = 1
     
    if A5_Score == "True now and when I was young":
        A5_Score = 0
    elif A5_Score == 'True only now':
        A5_Score = 0
    elif A5_Score == 'True when I was <16':
        A5_Score = 1
    else:
        A5_Score = 1
     
    if A6_Score == "True now and when I was young":
        A6_Score = 0
    elif A6_Score == 'True only now':
        A6_Score = 0
    elif A6_Score == 'True when I was <16':
        A6_Score = 1
    else:
        A6_Score = 1
    
    if A7_Score == "True now and when I was young":
        A7_Score = 0
    elif A7_Score == 'True only now':
        A7_Score = 0
    elif A7_Score == 'True when I was <16':
        A7_Score = 1
    else:
        A7_Score = 1

    if A8_Score == "True now and when I was young":
        A8_Score = 0
    elif A8_Score == 'True only now':
        A8_Score = 0
    elif A8_Score == 'True when I was <16':
        A8_Score = 1
    else:
        A8_Score = 1
  
    if A9_Score == "True now and when I was young":
        A9_Score = 0
    elif A9_Score == 'True only now':
        A9_Score = 0
    elif A9_Score == 'True when I was <16':
        A9_Score = 1
    else:
        A9_Score = 1
   
    if A10_Score == "True now and when I was young":
        A10_Score = 0
    elif A10_Score == 'True only now':
        A10_Score = 0
    elif A10_Score == 'True when I was <16':
        A10_Score = 1
    else:
        A10_Score = 1
   
               
diagnosis=''
if(st.button('Prediction results')):
    diagnosis= predict('A1_Score','A2_Score', 'A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score')

if __name__=="__main__":
    main()

           
 