# README for test.py
## Overview
This script, test.py, processes test data, loads a pre-trained model along with a preprocessing pipeline, makes predictions on dark matter halo concentration, and outputs the results. It also provides a visualization by plotting predicted values against actual labels, calculates the Root Mean Square Error (RMSE) score, and saves the predictions to a CSV file.

## Requirements
### Python Libraries
To run the script, you need the following Python libraries:

```numpy```
```pandas```
```joblib```
```matplotlib```
```sklearn```
```dill```
You can install these using the following command:

```bash
pip install numpy pandas joblib matplotlib scikit-learn dill
```

## Instructions
### Step 1: Set the Path to the Test Data
In the script, the variable test_data_path is already set to:
```python
test_data_path = r'F:\DM\full_data.csv'
```

Ensure that this path points to the correct CSV file on your machine.


### Step 2: Check for Missing or Extra Columns
The script automatically checks whether the test data has the required columns. If any expected columns are missing or extra columns are present, it will raise an error or drop the extra columns.

Expected columns:
```python
expected_columns = [
    'Virial Mass', 'Virial Radius', 'Velocity Disp', 'Vmax', 'Spin',
    'B to A', 'C to A', 'Energy ratio', 'Peak Mass', 'peak Vmax',
    'Halfmass a', 'Peakmass a', 'Acc Rate', 'Concentration'
]
```

### Step 3: Load the Preprocessing Pipeline and Model
Paths to the saved preprocessing pipeline and  model are already set in the script:
```python
pipeline_path = r'F:\DM\full_pipeline_1.pkl'  # Path to preprocessing pipeline
xgb_model_path = r'F:\DM\Model\best_xgb_model.pkl'  # Path to saved XGBoost model
```
Ensure these paths point to the correct locations on your machine where the files are stored.

### Step 5: Plot Predictions vs. Actual Labels
If your test data contains actual labels for the dark matter halo concentration, the script will generate a plot showing the predicted values versus the actual labels.

 - Diagonal Line: Represents a perfect prediction.
 - Dots: Represent the model’s predictions versus the actual values.

### Step 6: RMSE Calculation
The script calculates the Root Mean Squared Error (RMSE) between the predicted values and the actual labels. This gives an idea of how well the model performed.

The RMSE score is printed as follows:
```bash
The RMSE score of the prediction: <RMSE_Score>
```

### Step 7: Save Predictions to CSV
The predicted dark matter halo concentration values are saved to a CSV file named ```predictions_output.csv``` in the current working directory.
```bash
Predictions saved to predictions_output.csv
```

## Running the Script
To run the script, use the following command:
```bash
python test.py
```
Ensure that the test CSV file, preprocessing pipeline, and model paths are correctly set before running the script.

## Output
- Plot: A graph showing the predicted concentrations against actual labels.
- RMSE Score: The calculated RMSE score printed to the console.
- CSV File: A file named predictions_output.csv containing the predicted dark matter halo concentrations.

## Notes
- If your test data has missing columns, the script will raise an error indicating the missing features.
- The paths for test data, model pipeline, and XGBoost model are already set; ensure they match the locations on your system.
