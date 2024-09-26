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
 pip install fastapi uvicorn pandas numpy scikit-learn matplotlib pydantic dill

