# test.py
# Author: Rishav Bhattacharjee
# Team Members: Rishav Bhattacharjee, Gayatri Sonawane, Madhushree Ravichandan
# Date: 8th September, 2024
# This script handles the test data, loads a saved model pipeline, 
# processes the test data, makes predictions, and saves them to a CSV or plots them.

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.metrics import mean_squared_error
import dill

# Step 1: Load the test data provided by the user
test_data_path = r'F:\DM\full_data.csv'  # Replace with the actual test CSV path
test_data = pd.read_csv(test_data_path)

# Step 2: Handle missing or undefined values in the test data
test_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Check for missing columns compared to the training data
expected_columns = [
    'Virial Mass', 'Virial Radius', 'Velocity Disp', 'Vmax', 'Spin',
    'B to A', 'C to A', 'Energy ratio', 'Peak Mass', 'peak Vmax',
    'Halfmass a', 'Peakmass a', 'Acc Rate', 'Concentration'
]

missing_columns = [col for col in expected_columns if col not in test_data.columns]
extra_columns = [col for col in test_data.columns if col not in expected_columns]

if missing_columns:
    raise ValueError(f"Missing columns in test data: {missing_columns}")
if extra_columns:
    print(f"Extra columns found and will be dropped: {extra_columns}")
    test_data = test_data.drop(columns=extra_columns)

# Step 3: Impute missing values in the test set
#imputer = SimpleImputer(strategy='median')
#test_data_imputed = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns)

#print(test_data_imputed.columns)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values

# Step 4: Load the preprocessing pipeline and the XGBoost model
#pipeline_path = r'F:\DM\Pipeline\full_pipeline.pkl'  # The path to your saved preprocessing pipeline
pipeline_path = r'F:\DM\full_pipeline_1.pkl'  # The path to your saved preprocessing pipeline
xgb_model_path = r'F:\DM\Model\best_xgb_model.pkl' # The path to your saved XGBoost model

#xgb_model_path = r'F:\DM\final_xgb_model.pkl'



# Load the preprocessing pipeline
pipeline = joblib.load(pipeline_path)

# Load the XGBoost model
xgb_model = joblib.load(xgb_model_path)

# Step 5: Preprocess the test data using the pipeline
y_test = test_data["Concentration"].values
#drop_data = test_data_imputed.drop("Concentration")
test_data_prepared = pipeline.transform(test_data)

#print(pipeline)

# Step 6: Make predictions using the XGBoost model

predictions = xgb_model.predict(test_data_prepared)

# Step 7: Plot predictions vs labels (if labels are present in your test data)

y_test = test_data["Concentration"].values


plt.figure(figsize=(10, 6))

# Scatter plot of predictions vs labels
plt.plot(predictions, y_test, '.', label='Predictions vs Labels')

# Plot a diagonal line to represent perfect predictions
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='orange', label='Ideal Fit')

# Add labels and title
plt.xlabel('Predictions')
plt.ylabel('Labels')
plt.title('Predictions vs Labels')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

#Step 8: Calculation the RMSE Score
RMSE_Score = np.sqrt(mean_squared_error(y_test, predictions))
print("The RMSE score of the prediction:", RMSE_Score)

#Step 9: Output of the predictions to a csv file

output_df = pd.DataFrame(predictions, columns=['Predicted Concentration'])
output_df.to_csv("predictions_output.csv", index=False)
print("Predictions saved to predictions_output.csv")








###Step 3 removed
