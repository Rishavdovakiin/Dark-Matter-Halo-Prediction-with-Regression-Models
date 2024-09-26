# fastapi_app.py
# Author: Rishav Bhattacharjee
# Date: 8th September, 2024
# FastAPI app for handling test data, loading saved model pipeline,
# processing the test data, making predictions, and saving them as CSV or plotting.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import io
import os
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
import dill


# Initialize FastAPI app
app = FastAPI()

# Enable CORS (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://appfrontendpy-cggdjxzvjyk6rcjlzntzmu.streamlit.app/"],  # Update this with your frontend domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the DataFrameSelector class for selecting columns
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names]

# Paths to the model pipeline and the model (can be any model)
pipeline_path = r'F:\DM\Pipeline\full_pipeline_dill.pkl'  # Update path
model_path = r'F:\DM\Model\best_xgb_model_dill.pkl'  # This can be any model (XGBoost, CatBoost, etc.)
#model_path = r'F:\DM\Model\best_cat_model.pkl'
#model_path = r'F:\DM\Model\best_linreg_model_dill.pkl'

# Load the preprocessing pipeline and model at startup
try:
    # Load the pipeline using dill
    with open(pipeline_path, 'rb') as f:
        full_pipeline = dill.load(f)
    print("Pipeline loaded successfully")

    # Load the model using dill (generic)
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    print("Model loaded successfully")
    
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

# Expected columns in the dataset
expected_columns = [
    'Virial Mass', 'Virial Radius', 'Velocity Disp', 'Vmax', 'Spin',
    'B to A', 'C to A', 'Energy ratio', 'Peak Mass', 'peak Vmax',
    'Halfmass a', 'Peakmass a', 'Acc Rate', 'Concentration'
]

# Thread pool for CPU-bound tasks (like model predictions)
executor = ThreadPoolExecutor()

# Utility function to handle model predictions in a non-blocking way
async def make_predictions(model, data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, model.predict, data)

# Root route to avoid 404 error
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI app! Use /predict/ for predictions."}

# Global variable to store the plot image
img_bytes = None

# Path to save outputs on Desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_folder = os.path.join(desktop_path, "ModelOutputs")

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Route to handle CSV file uploads and prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    global img_bytes  # Declare global to store the plot image in memory

    # Step 1: Load the test data from the uploaded file
    contents = await file.read()
    
    try:
        test_data = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")
    
    # Step 2: Handle missing or undefined values in the test data
    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Check for missing and extra columns
    missing_columns = [col for col in expected_columns if col not in test_data.columns]
    extra_columns = [col for col in test_data.columns if col not in expected_columns]
    
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing columns in test data: {missing_columns}")
    
    if extra_columns:
        test_data = test_data.drop(columns=extra_columns)

    # Step 3: Impute missing values in the test data
    try:
        imputer = SimpleImputer(strategy='median')
        test_data_imputed = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error imputing missing values: {e}")

    # Step 4: Preprocess the test data using the pipeline
    try:
        test_data_prepared = full_pipeline.transform(test_data_imputed)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during data preprocessing: {e}")

    # Step 5: Make predictions using the model asynchronously
    try:
        predictions = await make_predictions(model, test_data_prepared)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")
    
    # Step 6: If labels are present in the test data, create the plot in-memory
    rmse_score = None
    if 'Concentration' in test_data.columns:
        y_test = test_data["Concentration"].values
        plt.figure(figsize=(10, 6))
        plt.plot(predictions, y_test, '.', label='Predictions vs Labels')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='orange', label='Ideal Fit')
        plt.xlabel('Predictions')
        plt.ylabel('Labels')
        plt.title('Predictions vs Labels')
        plt.legend()
        plt.grid(True)

        # Save the plot to a file in the ModelOutputs folder
        plot_path = os.path.join(output_folder, "predictions_plot.png")
        plt.savefig(plot_path)
        plt.close()

        # Save the plot to an in-memory bytes buffer
        img_bytes = io.BytesIO()
        with open(plot_path, "rb") as f:
            img_bytes.write(f.read())
        img_bytes.seek(0)

        # Calculate RMSE
        rmse_score = np.sqrt(mean_squared_error(y_test, predictions))
    
    # Step 7: Output predictions to a CSV file in the ModelOutputs folder
    output_df = pd.DataFrame(predictions, columns=['Predicted Concentration'])
    csv_output_path = os.path.join(output_folder, "predictions_output.csv")
    output_df.to_csv(csv_output_path, index=False)

    # Step 8: Return prediction results with paths to CSV and RMSE score (if applicable)
    response = {
        "message": "Prediction successful.",
        "csv_output": csv_output_path
    }

    if rmse_score is not None:
        response["RMSE_Score"] = rmse_score

    return response

# Endpoint to serve the plot as an image
@app.get("/plot/")
async def get_plot():
    global img_bytes  # Use the global img_bytes to get the image data
    if img_bytes:
        return StreamingResponse(img_bytes, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Plot not available")

# Endpoint to download the CSV file
@app.get("/download_csv/")
async def download_csv():
    csv_path = os.path.join(output_folder, "predictions_output.csv")
    if os.path.exists(csv_path):
        return FileResponse(csv_path, media_type="text/csv", filename="predictions_output.csv")
    raise HTTPException(status_code=404, detail="CSV file not found")

# Endpoint to download the plot
@app.get("/download_plot/")
async def download_plot():
    plot_path = os.path.join(output_folder, "predictions_plot.png")
    if os.path.exists(plot_path):
        return FileResponse(plot_path, media_type="image/png", filename="predictions_plot.png")
    raise HTTPException(status_code=404, detail="Plot file not found")

# Main entry point for running the app directly
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
