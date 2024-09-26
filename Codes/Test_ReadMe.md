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
