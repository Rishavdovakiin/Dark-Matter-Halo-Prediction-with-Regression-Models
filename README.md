## Dark Matter Halo Prediction Using Machine Learning Models
This repository contains the results and code from my research internship, focused on predicting dark matter halo concentration using various advanced Machine Learning (ML) models, including Linear regression, Forest, XGBoost, and CatBoost. The models were trained and evaluated on the Bolshoi simulation dataset, a large cosmological simulation widely used in astrophysical research. This project aims to compare the performance of different ML techniques in predicting dark matter halo properties and make these trained models easily accessible for future research.

## Key Features:
- **Model Implementation**: Code for training multiple ML models (AdaBoost, XGBoost, CatBoost) for predicting dark matter halo concentration.
- **Performance Comparison**: Comprehensive comparison of model performances using evaluation metrics such as cross-validation scores and feature importance analysis.

- **Preprocessing Pipeline**: Custom preprocessing steps for feature scaling based on their statistical characteristics, ensuring optimal performance.

- **API Integration**: The API.py script allows seamless access to the trained models, enabling researchers to input their data and obtain predictions.

- **Interactive Streamlit App**: The streamlit.py script provides an interactive web application using Streamlit, allowing users to test the models and visualize predictions without extensive coding knowledge.

- **Test Suite**: The test.py file contains unit tests to validate the accuracy and robustness of the ML pipeline, ensuring that it works as expected with new data.

## Contents:
- **Bolshoi Simulation Data**: Utilizes the dark matter halo properties from the Bolshoi simulation for training and evaluation.

- **Model Training**: Implementation of multiple boosting algorithms with hyperparameter tuning.

- **Model Comparison**: Performance metrics and results, helping to identify the most effective approach for halo concentration prediction.

- **API Usage**: Simple instructions to use the pre-trained models via the API for your own dark matter simulation datasets.

- **Streamlit Web App**: A user-friendly interface for accessing the model predictions and visualizing results in real-time.

## Future Work:
Researchers working with the Bolshoi simulation or similar dark matter halo datasets can use this repository to:

- Train new models on additional or updated simulation data.
- Apply the trained models to make predictions for their research.
- Use the Streamlit app for quick testing of different halo property datasets.

## Getting Started:
To get started with using the models, the API, or the Streamlit app, refer to the respective documentation in the repository for detailed instructions.
