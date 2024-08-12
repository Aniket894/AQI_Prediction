from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

class PredictionPipeline:
    def __init__(self, model_path, preprocessor_path):
        # Load the pre-trained model and preprocessor
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_data):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        # Transform the data using the preprocessor
        processed_data = self.preprocessor.transform(input_df)
        # Make predictions using the model
        predictions = self.model.predict(processed_data)
        return predictions.tolist()
    
# Initialize the prediction pipeline with model and preprocessor paths
prediction_pipeline = PredictionPipeline(
    model_path='artifacts/DecisionTreeRegressor_best_model.pkl',
    preprocessor_path='artifacts/preprocessor.pkl'
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Extract data from the form
    input_data = {
        'Date': request.form.get('Date'),  # Assuming date is provided as a string in the format expected by the model
        'Latitude': float(request.form.get('Latitude')),
        'Longitude': float(request.form.get('Longitude')),
        'PM2.5': float(request.form.get('PM2.5')),
        'PM10': float(request.form.get('PM10')),
        'NH3': float(request.form.get('NH3')),
        'SO2': float(request.form.get('SO2'))
    }

    # Make prediction
    predictions = prediction_pipeline.predict(input_data)
    
    return render_template('results.html', predictions=predictions)
    
if __name__ == '__main__':
    app.run(debug=True)
