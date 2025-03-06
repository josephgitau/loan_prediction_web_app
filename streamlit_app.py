# importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# page config
st.set_page_config(page_title='Loan Prediction App', layout='wide')

# Add app title and description
st.title("Loan Approval prediction")
st.write("Enter the applicant's information to predict loan approval status")


## Ml Modelling Part
url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv"
df = pd.read_csv(url).set_index('Unnamed: 0') # remove unnamed column 0

# Dispay sample data to the user
st.subheader("Sample Loan Data")
st.dataframe(df.head(3))

## Preprocess loan data
def preprocess_data(df):
    # fill in missing values
    # df.fillna(method='pad')  # Forward fill
    df.dropna(inplace=True)

    # encode our dataset
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Married'] = le.fit_transform(df['Married'])
    df['Education'] = le.fit_transform(df['Education'])
    df['Dependents'] = le.fit_transform(df['Dependents'])
    df['Self_Employed'] = le.fit_transform(df['Self_Employed'])
    df['Property_Area'] = le.fit_transform(df['Property_Area'])

    return df

# process the data
preprocessed_df = preprocess_data(df)

# display sample data 
st.subheader("Preprocessed Data")
st.table(preprocessed_df.head(3))

## Train our ML Model
X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
y = df['Loan_Status'].map({1:"Yes", 0:"No"})

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# instantiate model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# evaluate model
from sklearn.metrics import classification_report

st.text(classification_report(y_test, y_pred))
