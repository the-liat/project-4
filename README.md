# Uncovering Hidden Patterns: Predictive Analytics for Gun Violence in the U.S

<img width="994" alt="Screenshot 2023-06-06 at 1 12 31 AM" src="https://github.com/the-liat/project-4/assets/119654958/78cc60ef-fca0-418f-b28e-43244f254e7f">

## Table of Contents: 
- [Overview](#overview)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [References](#references)

## Overview: 
The number of mass shootings in the US is increasing. This year alone, as of late May, the Gun Violence Archive (an online independent data collection and research group) has counted more than 260 mass shootings in the United States. The objective is to build a prediction model that assess the number deaths/incidents of gun violence in the U.S by county.

## Usage:
The application was built using Streamlit. The user can interact with the app. Here are the steps to launching the app using Streamlit:

1. Install Streamlit by running the following command `pip install streamlit` in your terminal or command prompt.
2. On a MAC you also need to install Pydrive by using the following command: `pip install pydrive`. 
3. Run the app locally by navigating to the directory where `st_app.py` is saved. Use the following command: `streamlit run st_app.py`. This will start the Streamlit development server, and you'll see a URL (e.g., http://localhost:8501) where your app is running. Open the URL in your web browser to view and interact with the app.

## Methodology: 

 **The code with comments can be found in the Notebooks file:** *Gun_violence_analysis_GroupCopy01.ipynb & Mass_shootings_forecastARIMA.ipynb.*

The data is from various sources. The sources include CSV files, Excel and web APIs. The data is cleaned using python. Feature engineering is performed to create meaningful features for the model. The cleaned data is stored in CSV format. Pandas DataFrame is used to import the data into the modeling environment.

Three models are evaluated: random forest classifier, sequential neural network, and ARIMA. Among these models, only the random forest classifier is used for the app. The DataFrame used for both the random forest classifier and sequential neural network consists of 38 total columns, with 28 columns representing features. The dataset contains 3104 rows, corresponding to 3143 counties in the U.S., and the target variable is death by gun.

The county features included in the dataset are unemployment rate, population, gun laws, number of chain restaurants, number of places of worship, poverty, education, crime, and race.

For the random forest model, the model is trained using the prepared training data. A binary classifier is constructed to predict whether gun deaths occur `(1)` or do not occur `(0)` in each county. The performance metrics and evaluation of this model can be found in the [results](#results) section of this documentation. Based on the binary model, the probabilities for each county are calculated, specifically the probability of at least 14 deaths by gun occurring in a county per year

For the sequential neural network, the model is also trained using the prepared training data. The neural network architecture is implemented using the Keras library. A binary neural network model is fined-tuned and created to predict whether deaths occurred `(1)` or did not occur `(0)` for a given county. The performance metrics and evaluation of this model can be found in the [results](#results) section of this documentation.

## Results: 

### ARIMA model
The ARIMA was employed in an effort to make forecasts on mass shooting incidents in the usa over the year. Given limitations with the data, specifically a) not enough observations (the the model requires a minimum of 60) and b) discontinuity (events do not occur on a consistent timetable that is required for the model), a decision was made to asses the available data in different ways.

![image](https://github.com/the-liat/project-4/assets/8871346/6192f61c-d310-4bf7-8c2a-ac51f6a37197)

### Random Forest Classifier 
<img width="387" alt="image" src="https://github.com/the-liat/project-4/assets/119654958/29eb92f1-f286-48c9-bee7-21cc9a7f1a8e">

The random forest classifier achieved an accuracy score of 0.906, correctly predicting deaths occurred `(1)` or did not occur `(0)` for about 90.6% of instances.

### Sequential Neural Network
<img width="406" alt="image" src="https://github.com/the-liat/project-4/assets/119654958/a4aa6121-0017-4782-b22a-b2f9e070cbeb">

The sequential neural network achieved an accuracy score of 0.898, correctly predicting deaths occurred `(1)` or did not occur `(0)` for about 89.8% of instances.

Both of these model results in reletavle similar accuracy. Precision, recall and f1-scores were also similar and in the 90% range dimming the models appropriate for predicting gun deaths in usa counties with good accuracy.





![image](https://github.com/the-liat/project-4/assets/119654958/08242ca9-af8a-48f6-98b4-5f496f96aaee)

![image (1)](https://github.com/the-liat/project-4/assets/119654958/8981242d-7d13-4d12-8875-a3176025535d)

![image (2)](https://github.com/the-liat/project-4/assets/119654958/14ecee04-cb86-43da-b5b3-d181094baebe)



### Limitations:

Missing Values: Dealing with missing values in the dataset (e.g., Hawaii, counties, and number of deaths) created gaps in the data. This can affect the model's ability to capture patterns and make accurate prediction. Strategies such as imputation or removal of missing values were employed to address this challenge.

Lack of Continuity in the Date Timeline: The time series data used for predicting gun violence had gaps or lacked continuity in the date timeline. This posed challenges for modeling with ARIMA, as it assumes a continuous and evenly spaced time series. The ARIMA model struggled to capture accurate time-dependent patterns and correlations in the presence of such gaps.

### Conclusion: 
In conclusion, the ARIMA model not suitable for our dataset. The random forest and neural network models demonstrate similar performance in predicting gun violence/deaths based on the dataset. The accuracy, precision, recall, and F1-scores indicate that the models are effective in distinguishing between cases where deaths occurred and cases where deaths did not occur. The predictions can be used to identify areas with higher risk of gun violence deaths, allowing policymakers, law enforcement agencies, and community organizations to allocate resources effectively and implement targeted strategies to address the problem. For future work, factors like mental health resources, access to firearms, social programs, or community engagement initiatives can be considered. They may enhance the model's predictive power and provide a better understanding of gun violence dynamics. 

### References: 
- State Firearm Law Database: State Firearm Laws, 1991-2019 (https://catalog.data.gov/dataset/state-firearm-law-database-state-firearm-laws-1991-2019-e2e9d)
- Gun Violence Archive (https://www.gunviolencearchive.org/)
- Gun Deaths by County (https://data.world/nkrishnaswami/gun-deaths-by-county)
- Firearm regulations in the U.S. (https://www.kaggle.com/code/jonathanbouchet/firearm-regulations-in-the-u-s) 
- Gun Violence, USA (https://www.kaggle.com/datasets/nidzsharma/us-mass-shootings-19822023) 
- The Violence Project (https://www.theviolenceproject.org/)
- Education (https://data.world/usda/county-level-data-sets)
- Unemployment Rate (https://data.world/usda/county-level-data-sets)
- Poverty (https://data.world/usda/county-level-data-sets)
- Number of Chain Restaurants (https://www.kaggle.com/datasets/michaelbryantds/us-restaurants)
- Gunlaws (https://www.kaggle.com/datasets/michaelbryantds/us-restaurants)
- Race 2010-2019) https://data.world/bdill/county-level-population-by-race-ethnicity-2010-2019
- Places of Worship (2020) https://hifld-geoplatform.opendata.arcgis.com/datasets/geoplatform::all-places-of-worship/about & https://data.world/niccolley/us-zipcode-to-county-state
- Crime (https://www.openicpsr.org/openicpsr/project/115006/version/V1/view?path=/openicpsr/115006/fcr:versions/V1.4/nanda_crime_county_2002-2014_01P_csv_with_readme.zip&type=file)


