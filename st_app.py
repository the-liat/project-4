# importing dependencies

import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os
import hvplot.pandas

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import folium
from folium.plugins import MarkerCluster
from folium import FeatureGroup

import numpy as np

import warnings
warnings.filterwarnings('ignore')

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Getting the combined dataset
#-----------------------------------------------------------------------------------
@st.cache_data
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
st.write("## Gun Violence in the United States")

# displaying dataset in app 
if st.checkbox('**Show dataset**'):
    counties_gvd

#-----------------------------------------------------------------------------------

# creating line chart
st.write("#### Line Chart Title")

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

# creating a map
st.write("#### Map of San Francisco")

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

# adding a slider
st.write("#### Example of a slider")

x = st.slider('Slide the bar to select a number')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

# requesting user input
st.write("#### Example of text input")

st.text_input("Enter your name:", key="name")
st.session_state.name #accessing the value

# using checkboxes to show/hide data
st.write("#### Example of a Checkbox")

if st.checkbox('**Show dataframe**'):
    df2 = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    df2

# using a selectbox for options
st.write("#### Example of a Dropdown Menu (selectbox)")
option = st.selectbox(
    'Which number do you like best?',
     [1,2,3,7,8,4])

'You selected: ', option

# Layout: pinning widgets to a left panel sidebar
#################################################

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

"""Beyond the sidebar, Streamlit offers several other ways to control 
the layout of your app. st.columns lets you place widgets side-by-side, 
and st.expander lets you conserve space by hiding away large content."""

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")

# ------------------------------------------------------------------------------

# Random forest model
#####################################################################


# create a copy of counties_gvd
copy_counties = counties_gvd.copy()

# Convert the target variable to binary, '0' for no deaths, all else, '1'
X = copy_counties
X['deaths_occurred_gt1'] = (X['deaths_per_100K_mean (1999-2020)'] != 0).astype(int)

# Define the target
y = X['deaths_occurred_gt1']

# Define features set
X.drop(['Unnamed: 0', 'county_Abbr', 'county_code', 'state', 'Area name', 'county', 
        'Abbr', 'deaths_per_100K_mean (1999-2020)', 'deaths_per_100K_std (1999-2020)', 
        'deaths_per_100K_numb_years (1999-2020)', 'deaths_occurred_gt1', 'name', 'lng', 'lat'], 
       axis=1, inplace=True)
""" for the app instead of the X.drop above, create a list of features - as checkboxes
and these will be entered into the model in real time. 
The app will display the number of features selected the accuracy and the confusion matrix  

['growth_rate_gun_laws (1991-2017)',
 'avg_pop (2010-2019)',
 'avg_murder (2002-2014)',
 'avg_rape (2002-2014)',
 'avg_robbery (2002-2014)',
 'avg_agasslt (2002-2014)',
 'avg_burglry (2002-2014)',
 'avg_larceny (2002-2014)',
 'avg_mvtheft (2002-2014)',
 'avg_arson (2002-2014)',
 'Poverty % overall (2014)',
 'Poverty % ages 0-17 (2014)',
 'Poverty % related kids age 5-17 (2014)',
 'Percent of adults with less than a high school diploma, 2010-2014',
 'Percent of adults with a high school diploma only, 2010-2014',
 "Percent of adults completing some college or associate's degree, 2010-2014",
 "Percent of adults with a bachelor's degree or higher, 2010-2014",
 'avg_unemp_rate_2010_2015',
 '% white pop avg (2010-2019)',
 '% black pop avg (2010-2019)',
 '% asian pop avg (2010-2019)',
 '% indian pop avg (2010-2019)',
 '% pacific pop avg (2010-2019)',
 '% two pop avg (2010-2019)',
 '% not hisp pop avg (2010-2019)',
 '% hisp pop avg (2010-2019)',
 'chain restaurants per 100K',
 'places of worship per 100K']
"""



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

# Calculating the confusion matrix
cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
)

# Calculating the accuracy score
acc_score = accuracy_score(y_test, predictions)

# Displaying results
print("Confusion Matrix")
print("~" * 30)
display(cm_df)
print("-" * 70)
print(f"Accuracy Score : {acc_score}")
print("-" * 70)
print()
print("Classification Report")
print("~" * 30)
print(classification_report(y_test, predictions))