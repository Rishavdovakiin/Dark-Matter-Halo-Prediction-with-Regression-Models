# FastAPI Model Prediction App

Author: Rishav Bhattacharjee  
Date: 8th September, 2024

This FastAPI application allows users to upload a test dataset, which is then processed using a pre-trained machine learning model. The app supports CSV uploads, performs data preprocessing, makes predictions using the model, and provides options for downloading the results as a CSV file or visualizing them as a plot.

## Key Features:
1. **Upload CSV**: Upload test data in CSV format.
2. **Prediction**: Preprocess the data and make predictions using a pre-trained model pipeline.
3. **Visualization**: Generate and download a plot comparing predictions to actual values (if labels are provided).
4. **CSV Output**: Save and download the prediction results as a CSV file.
5. **Cross-Origin Resource Sharing (CORS)**: Allows communication with other domains.

---

## Instructions for Running the App

### Prerequisites:
1. **Python 3.7+** installed on your system.
2. Install the required dependencies using pip:\
 # pip install fastapi uvicorn pandas numpy scikit-learn matplotlib pydantic dill


3. **Model Files**: You need the following pre-trained model and pipeline files saved on your system:
- Preprocessing pipeline (`full_pipeline_dill.pkl`)
- Pre-trained machine learning model (e.g., `best_xgb_model_dill.pkl`)

Make sure to update the paths to these files in the `pipeline_path` and `model_path` variables in `fastapi_app.py`.

### Running the FastAPI App:
1. Clone or copy this project to your local machine.
2. Ensure the required model and pipeline files are present and the paths in the code are updated correctly.
3. Open a terminal, navigate to the project directory, and run the following command:
# uvicorn fastapi_app --reload

