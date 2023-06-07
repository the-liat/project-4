# importing dependencies
# ------------------------------------------------------------------------------
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import numpy as np

import warnings
warnings.filterwarnings('ignore')

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Getting the combined dataset
#-----------------------------------------------------------------------------------
@st.cache_data(persist=None)
def get_data(url, name):
    # download the file from the Google Drive link
    gdown.download(url, name, quiet=False)
    # Read the file into a dataframe
    df = pd.read_csv(name)
    return df

# define the Google Drive file URL and file name
file_url = 'https://drive.google.com/uc?id=1N_hr60DCdSRyjmzerZlFeg8LGMmlbYUs'
file_name = 'counties_gvd.csv'

# use the function to get the data and define the dataframe
counties_gvd = get_data(file_url, file_name)

# creating app header
st.title("Gun Violence in the :blue[United] :red[States]")

# displaying dataset in app 
if st.checkbox('**Show dataset**'):
    counties_gvd

#####################################################################
# Random forest model
#####################################################################

# Setting up the features and display on the app
#-----------------------------------------------------------------------------------
st.header(":blue[Random Forest Model]")

# create a copy of counties_gvd
copy_counties = counties_gvd.copy()

# Convert the target variable to binary, '0' for no deaths, all else, '1'
X = copy_counties
X['deaths_occurred_gt1'] = (X['deaths_per_100K_mean (1999-2020)'] != 0).astype(int)

# Define the target
y = X['deaths_occurred_gt1']

# Display target name on app
st.write("#### Target: Predicting whether or not death occured in the county")
st.caption("Note: original variable was the average number of deaths per 100,000 people between 1999 and 2000")
st.divider()

# Define features set
X.drop(['Unnamed: 0', 'county_Abbr', 'county_code', 'state', 'Area name', 'county', 
        'Abbr', 'deaths_per_100K_mean (1999-2020)', 'deaths_per_100K_std (1999-2020)', 
        'deaths_per_100K_numb_years (1999-2020)', 'deaths_occurred_gt1', 'name', 'lng', 'lat'], 
       axis=1, inplace=True)

# Display names of features on app
@st.cache_data(persist=None)
def show_features():
    col1, col2, col3 = st.columns(3)
    with col1:
        features_1 = ['Growth Rate in Gun Laws', 
                    'Average Population Size',
                    'Average Rape Rate',
                    'Average Roberry Rate',
                    'Average Assualt Rate',
                    'Average Burglry Rate',
                    'Average Larceny Rate',
                    'Average Theft Rate',
                    'Average Arson Rate']
        for feat in features_1:
         st.write(feat)
         
    with col2:
       features_2 = ['Poverty Percent Overall',
                 'Poverty Percent Ages 0-17',
                 'Poverty Percent related kids age 5-17',
                 'Percent of Adults with < HS diploma',
                 'Percent of Adults with HS diploma',
                 'Percent of Adults with some college',
                 'Percent of Adults with BA or higher',
                 'Number of Chain Restaurants per 100K',
                 'Number of Places of Worship per 100K']
       for feat in features_2:
          st.write(feat)
          
    with col3:
       features_3 = ['Average Unemployment Rate',
                  'Average Percent of White Population',
                  'Average Percent of Black Population',
                  'Average Percent of Asian Population', 
                  'Average Percent of Indian Population', 
                  'Average Percent of Pacific Population', 
                  'Average Percent of Two+ Races Population',
                  'Average Percent of Not-Hispanic Population',
                  'Average Percent of Hispanic Population']
       for feat in features_3:
          st.write(feat)

if st.checkbox('**Show List of Features**'):
   show_features()
st.divider()

# Creating the model
#-----------------------------------------------------------------------------------
@st.cache_data(persist=None)
def rd_model(X, y):
    # Splitting into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

    # Creating StandardScaler instance
    scaler = StandardScaler()

    # Fitting Standard Scaller
    X_scaler = scaler.fit(X_train)

    # Scaling the data
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Create a random forest classifier
    rf_model = RandomForestClassifier(n_estimators=500, random_state=78)

    # Fitting the model
    rf_model = rf_model.fit(X_train_scaled, y_train)

    # Making predictions using the testing data
    predictions = rf_model.predict(X_test_scaled)

    # Creating predictions for the train data
    predictions2 = rf_model.predict(X_train_scaled)

    # Calculate feture importance and convert into a pandas DataFrame
    importances_df = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns), reverse=True))
    importances_df.set_index(importances_df[1], inplace=True)
    importances_df.drop(columns=1, inplace=True)
    importances_df.rename(columns={0: 'Feature Importances'}, inplace=True)
    importances_sorted = importances_df.sort_values(by='Feature Importances')
 
    return y_test, predictions, predictions2, y_train, X_train, X_test, importances_sorted

# Displaying Model Results
#-----------------------------------------------------------------------------------

# calling the random forest function
y_test, predictions, predictions2, y_train, X_train, X_test, importances_sorted = rd_model(X, y)

# Calculating the confusion matrix
cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
)

# Calculating the accuracy score
acc_score = accuracy_score(y_test, predictions)

# Displaying results
st.subheader(":blue[Model Results]")
st.write("#### :red[Confusion Matrix]")
st.dataframe(cm_df)
st.divider()

# Restructuring classification report to display correctly on the streamlit app
@st.cache_data(persist=None)
def class_report(y_test, predictions):
    results = pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose()
    results['precision'] = results['precision'].astype(float).map("{:.3f}".format)
    results['recall'] = results['recall'].astype(float).map("{:.3f}".format)
    results['f1-score'] = results['f1-score'].astype(float).map("{:.3f}".format)
    results.loc["accuracy", ['precision', 'recall']] = "-"
    results.loc["accuracy", 'support'] = results.loc["macro avg", 'support']
    results['support'] = results['support'].astype(int)
    return results
    
st.write("#### :red[Classification Report]")
st.dataframe(class_report(y_test, predictions))
st.divider()

# Feature importance bar chart
#-----------------------------------------------------------------------------------
st.write("### :red[Top Ten Important Features in Predicting Death in Counties]")
fig, ax = plt.subplots()
ax = importances_sorted[-10:].plot(kind='barh', color='blue', title= 'Features Importances: Top Ten', legend=False)
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
st.divider()


# Setting up prediction dataframe for interactive app
#-----------------------------------------------------------------------------------
@st.cache_data(persist=None)
def create_prediction_df(counties_gvd):
    # Create a copy of counties_gvd
    predictions_df = counties_gvd.copy()

    # Add the combined y values
    predictions_df['Actuals: Death Y/N'] = pd.concat([y_train, y_test])

    # Add the combined predictions
    predictions_df['Predictions: Death Y/N'] = pd.concat(
    [pd.Series(predictions2, index=X_train.index),
        pd.Series(predictions, index=X_test.index)])

    # Select the desired columns for predictions_df
    predictions_df = predictions_df[['state', 'county', 'Actuals: Death Y/N', 'Predictions: Death Y/N']]

    # Determining whether predictions are accurate (1) or not (0)
    predictions_df['Number of Accurate Predictions']= \
        1-(predictions_df['Predictions: Death Y/N']-predictions_df['Actuals: Death Y/N']).abs()
    
    return predictions_df

predictions_df = create_prediction_df(counties_gvd)

# Setting up summary dataframe for interactive app (accuracy per state)
#-----------------------------------------------------------------------------------
@st.cache_data(persist=None)
def create_summary_df(predictions_df):
    # group dataframe by state
    sum_df = predictions_df.groupby('state').sum().reset_index()
    count_county = predictions_df.groupby('state').count().reset_index()['county']

    # Join the sum_df and count_county DataFrames by index (state)
    combined_df = sum_df.join(count_county)

    # Rename the column for count_county
    combined_df.rename(columns={'county': 'Number of Counties'}, inplace=True)

    # droping the actuals and predictions, leaving just number of counties and number of accurate predictions 
    combined_df.drop(['Actuals: Death Y/N', 'Predictions: Death Y/N'], axis=1, inplace=True)

    # adding % accuracy column
    combined_df['Percent Accurate Predicitions'] = \
    round(combined_df['Number of Accurate Predictions']/combined_df['Number of Counties']*100, 2)

    return combined_df

combined_df = create_summary_df(predictions_df)

# Displaying predictions on the app
#-----------------------------------------------------------------------------------
st.write("#### :red[Predictions]")

# Define options for the dropdown
states =  combined_df['state'].sort_values().to_list()

# Add a select component
selected_state = st.selectbox('Select a state:', states)

# display on app
st.write(f"Here are the predictions by {selected_state}'s counties:")

selected_values = predictions_df.loc[predictions_df['state'] == selected_state, 
                                     ['county', 'Actuals: Death Y/N', 'Predictions: Death Y/N']]

st.dataframe(selected_values)

st.write("Here is the summary of predictions for the state of", selected_state)

summary_values = combined_df.loc[combined_df['state'] == selected_state, 
                                 ['Number of Accurate Predictions', 'Number of Counties', 'Percent Accurate Predicitions']]

st.dataframe(summary_values)

st.divider()
if st.checkbox(":green[_Select if you're done exploring the model_]"):
   st.balloons()