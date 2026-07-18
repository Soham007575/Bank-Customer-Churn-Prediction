import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
 
import tensorflow as tf
import pickle

model=tf.keras.models.load_model('model.h5')

with open( 'lable_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pickle','rb') as file:
    scaler=pickle.load(file) 



st.title('Customer Churn Prediction')

CreditScore = st.number_input("Credit Score", min_value=0, max_value=1000, step=1)

Geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])

Gender = st.selectbox("Gender",label_encoder_gender.classes_)

Age = st.slider("Age", 18,92)

Tenure = st.number_input("Tenure ",0,10 )

Balance = st.number_input("Balance")

NumOfProducts = st.slider("Number of Products", 1,4)

HasCrCard = st.selectbox("Has Credit Card", [0, 1])

IsActiveMember = st.selectbox("Is Active Member", [0, 1])

EstimatedSalary = st.number_input("Estimated Salary")

input_data = pd.DataFrame( {
    "CreditScore": [CreditScore],
  
    "Gender": [label_encoder_gender.transform([Gender])[0]],
    "Age": [Age],
    "Tenure": [Tenure],
    "Balance": [Balance],
    "NumOfProducts": [NumOfProducts],
    "HasCrCard": [HasCrCard],
    "IsActiveMember": [IsActiveMember],
    "EstimatedSalary": [EstimatedSalary]
})

geo_encoded=onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data,geo_encoded_df],axis=1)


input_data_scaled=scaler.transform(input_data)


prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]

st.write(f"Prediction probability : {prediction_prob:.2f}")

if prediction_prob > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")    