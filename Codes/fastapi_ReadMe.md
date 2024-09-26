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
```bash 
pip install fastapi uvicorn pandas numpy scikit-learn matplotlib pydantic dill
```

3. **Model Files**: You need the following pre-trained model and pipeline files saved on your system:
- Preprocessing pipeline (`full_pipeline_dill.pkl`)
- Pre-trained machine learning model (e.g., `best_xgb_model_dill.pkl`)

Make sure to update the paths to these files in the `pipeline_path` and `model_path` variables in `fastapi_app.py`.

### Running the FastAPI App:
1. Clone or copy this project to your local machine.
2. Ensure the required model and pipeline files are present and the paths in the code are updated correctly.
3. Open a terminal, navigate to the project directory, and run the following command:
uvicorn fastapi_app --reload

4. The app will be hosted on `http://127.0.0.1:8000/`.

---

## API Endpoints

### 1. Root (`/`)
- **Method**: `GET`
- **Description**: Basic root route that returns a welcome message.
- **Response**: 
```json
{
 "message": "Welcome to the FastAPI app! Use /predict/ for predictions."
}
```
## 2. Predict (/predict/)
- **Method**:`POST`
- **Description**: Upload a CSV file to make predictions. The file must contain test data with expected columns. Missing values and preprocessing are handled automatically.
- **Response**: A JSON object containing the path to the saved CSV output file and, if applicable, the RMSE score.
- **Request**: Upload a CSV file with the expected columns.
- **Example**:
```bash
curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@path_to_your_test_data.csv"
```

### 3. Plot (/plot/)
- **Method**: GET
- **Description**: Fetch the generated plot (only available if labels are provided in the test data).
- **Response**: Returns a PNG image comparing predictions vs labels.
- 
### 4. Download CSV (/download_csv/)
- **Method**: GET
- **Description**: Download the CSV file containing predictions.
- **Response**: Returns a CSV file with predicted concentration values.
### 5. Download Plot (/download_plot/)
- **Method**: GET
- **Description**: Download the prediction plot as a PNG image.
- **Response**: Returns a PNG image file of the plot (if available).

### Folder Structure
- `fastapi_app.py`: Main FastAPI application code.
- `ModelOutputs/`: A folder where the app will save the prediction CSV and plot files.\
  - The files will be stored on the desktop under `ModelOutputs`.

### Troubleshooting
- **CORS Issues**: If you're facing CORS-related issues while connecting from a frontend app, ensure you update the CORS settings under allow_origins.
- **File Upload Errors**: Ensure that the CSV file is correctly formatted and contains all the required columns.
- **Model Load Errors**: Double-check the paths and ensure that the correct model files are being loaded.

# Enjoy using the FastAPI Prediction App!

```css
This readme provides a step-by-step guide for setting up and running your FastAPI application, covering all necessary details from installation to endpoints and troubleshooting.
```
