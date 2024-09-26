README: FastAPI Model Prediction App
Author: Rishav Bhattacharjee
Date: 8th September, 2024

Overview
This FastAPI application allows users to upload test data, run it through a pre-trained machine learning model pipeline, and get predictions. It also provides the functionality to download the prediction results as a CSV file and visualize the predictions versus actual labels (if available) in a plot.

Features
Upload CSV File: Users can upload a CSV file containing test data.
Model Prediction: The app loads a pre-trained model and pipeline to process the uploaded data and make predictions asynchronously.
Downloadable Results: The prediction results are saved as a CSV file, which can be downloaded.
Plotting: If the true labels ('Concentration') are present in the data, a plot showing the predictions vs. true labels is generated and can be downloaded.
CORS: Cross-Origin Resource Sharing (CORS) is enabled to allow requests from different domains.
Setup Guide
Prerequisites
Python (version 3.8 or higher)

Virtual Environment (optional, but recommended)

Install the necessary Python packages using pip:

bash
Copy code
pip install fastapi uvicorn numpy pandas scikit-learn matplotlib dill
Folder Structure
The code assumes the following folder structure:

makefile
Copy code
F:\
 └── DM\
     ├── Pipeline\
     │   └── full_pipeline_dill.pkl   # Preprocessing pipeline file
     └── Model\
         └── best_xgb_model_dill.pkl  # Trained model file
Modify the paths in the code if your files are located elsewhere.

Running the App
Activate virtual environment (if you have set one up):

bash
Copy code
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows
Start the FastAPI server using Uvicorn:

bash
Copy code
uvicorn fastapi_app:app --reload
Access the app:

Open your browser and navigate to http://127.0.0.1:8000/
This will display a welcome message and confirm the app is running.
Endpoints
1. Root Endpoint
URL: /
Method: GET
Description: Returns a welcome message.
2. Predict Endpoint
URL: /predict/
Method: POST
Description: Accepts a CSV file containing test data, processes it, makes predictions, and saves the results as a CSV.
Input: UploadFile - CSV file with expected columns.
Output: JSON response with the path to the saved CSV file and the RMSE score (if applicable).
Expected Columns in the CSV:

text
Copy code
'Virial Mass', 'Virial Radius', 'Velocity Disp', 'Vmax', 'Spin', 
'B to A', 'C to A', 'Energy ratio', 'Peak Mass', 'peak Vmax', 
'Halfmass a', 'Peakmass a', 'Acc Rate', 'Concentration'
3. Plot Endpoint
URL: /plot/
Method: GET
Description: Serves the plot image (predictions vs actual labels) as a streaming response if labels are available in the dataset.
4. Download CSV Endpoint
URL: /download_csv/
Method: GET
Description: Allows downloading the prediction results as a CSV file.
5. Download Plot Endpoint
URL: /download_plot/
Method: GET
Description: Allows downloading the generated plot as a PNG image (if created).
Output Files
CSV Output File: The predictions are saved as predictions_output.csv in a folder on your Desktop (~/Desktop/ModelOutputs/).
Plot Image: If generated, the plot is saved as predictions_plot.png in the same folder.
Error Handling
Missing or Extra Columns: If the uploaded CSV file does not match the expected columns, an error will be raised.
Invalid CSV Format: If the file format is incorrect, a 400 HTTP error is returned.
Missing Plot or CSV: If the plot or CSV file is unavailable when requested, a 404 error will be raised.
Example Request Using curl
To upload a CSV file for prediction:

bash
Copy code
curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@your_test_data.csv"
Dependencies
This app uses the following Python libraries:

FastAPI: For building the web API.
Uvicorn: ASGI server for running FastAPI.
Pandas: For handling CSV files and data frames.
NumPy: For numerical computations.
Scikit-learn: For imputation, preprocessing, and model prediction.
Matplotlib: For plotting predictions vs actual labels.
Dill: For loading the saved model and preprocessing pipeline.
io, os, tempfile: For file I/O operations.
asyncio: For handling asynchronous tasks.
Future Improvements
Additional Model Support: The app could be expanded to allow users to choose from multiple models.
Dynamic Column Validation: Allow dynamic handling of columns so the app can support different datasets with ease.
Frontend Integration: Add a simple frontend to make it easier for users to upload files and view results.
