# streamlit_frontend.py
# Author: Rishav Bhattacharjee
# Date: 9th September, 2024
# Streamlit frontend for interacting with the FastAPI model prediction API.

import streamlit as st
import requests
import pandas as pd
from io import StringIO

# Set the FastAPI backend URL
API_URL = "http://127.0.0.1:8000"  # Update this if running on a different host/port

# Add custom CSS to set a galaxy background image (you can replace the URL with your own image)
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://c4.wallpaperflare.com/wallpaper/681/554/339/abstract-planet-space-purple-wallpaper-preview.jpg");
    background-size: cover;
    background-position: center;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title of the app
st.title("Dark Matter Halo Concentration Prediction")
st.write("Upload your CSV file to make predictions and visualize results.")

# File uploader for CSV files
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Display a preview of the uploaded CSV file
if uploaded_file is not None:
    st.write("Preview of the uploaded file:")
    file_contents = uploaded_file.getvalue().decode("utf-8")
    test_data = pd.read_csv(StringIO(file_contents))
    st.write(test_data)

    # Trigger the prediction when the "Predict" button is clicked
    if st.button("Predict"):
        with st.spinner("Making predictions..."):
            try:
                # Step 1: Send the CSV file to FastAPI backend for predictions
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{API_URL}/predict/", files=files)

                # Step 2: Handle the response from FastAPI
                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction completed successfully!")

                    # Step 3: Display RMSE score if available
                    if "RMSE_Score" in result:
                        st.write(f"RMSE Score: {result['RMSE_Score']}")

                    # Step 4: Display download button for CSV output
                    csv_output_path = result.get("csv_output")
                    if csv_output_path:
                        st.write("Download the predicted results:")
                        csv_download_link = f"{API_URL}/download_csv"
                        st.download_button(
                            label="Download CSV",
                            data=requests.get(csv_download_link).content,
                            file_name="predictions_output.csv",
                            mime="text/csv"
                        )

                    # Step 5: Display predictions vs labels plot if available
                    plot_output_url = f"{API_URL}/download_plot"
                    try:
                        image_data = requests.get(plot_output_url).content
                        st.image(image_data, caption="Predictions vs Labels", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error while fetching plot: {e}")
                else:
                    # Step 6: Handle errors returned by the backend
                    st.error(f"Error from backend: {response.json()['detail']}")

            except Exception as e:
                st.error(f"Error while making predictions: {e}")
