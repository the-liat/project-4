# Uncovering Hidden Patterns: Predictive Analytics for Gun Violence in the U.S

## Table of Contents: 
- Overview
- Usage
- Methodology
- Results
- Limitations and Future Work
- Conclusion
- References/Data Sources

## Overview: 
The number of mass shootings in the US is increasing. This year alone, as of late May, the Gun Violence Archive (an online independent data collection and research group) has counted more than 260 mass shootings in the United States. The objective is to build a prediction model that assess the number deaths/incidents of gun violence in the U.S by county.

## Installation Instructions/Usage:
The application was built using Streamlit. The user can interact with the app. Here are the steps to launching the app using Streamlit:

1. Install Streamlit by running the following command `pip install streamlit` in your terminal or command prompt.
2. Install Pydrive by using the following command: `pip install pydrive`. 
3. Run the app locally by navigating to the directory where `st_app.py` is saved. Use the following command: `streamlit run `st_app.py`. This will start the Streamlit development server, and you'll see a URL (e.g., http://localhost:8501) where your app is running. Open the URL in your web browser to view and interact with your app.

## Methodology: 
-	The Models 

## Results: 

### Random Forest Classifier 
<img width="387" alt="image" src="https://github.com/the-liat/project-4/assets/119654958/29eb92f1-f286-48c9-bee7-21cc9a7f1a8e">

### Sequential Neural Network
<img width="406" alt="image" src="https://github.com/the-liat/project-4/assets/119654958/a4aa6121-0017-4782-b22a-b2f9e070cbeb">

## Limitations and Future Work:

Missing Values: Dealing with missing values in the dataset (e.g., Hawaii, counties, and crime data) created gaps in the data. This can affect the model's ability to capture patterns and make accurate prediction. Strategies such as imputation or removal of missing values were employed to address this challenge.

Lack of Continuity in the Date Timeline: The time series data used for predicting gun violence had gaps or lacked continuity in the date timeline. This posed challenges for modeling with ARIMA, as it assumes a continuous and evenly spaced time series. The ARIMA model struggled to capture accurate time-dependent patterns and correlations in the presence of such gaps.

## Conclusion: 
Both the random forest and neural network models demonstrate similar performance in predicting gun violence/deaths based on the dataset. The accuracy, precision, recall, and F1-scores indicate that the models are effective in distinguishing between cases where deaths occurred and cases where deaths did not occur. The predictions can be used to identify areas with higher risk of gun violence deaths, allowing policymakers, law enforcement agencies, and community organizations to allocate resources effectively and implement targeted strategies to address the problem. For future work, factors like mental health resources, access to firearms, social programs, or community engagement initiatives can be considered. They may enhance the model's predictive power and provide a better understanding of gun violence dynamics. 

## References/Data Sources: 

