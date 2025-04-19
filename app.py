from flask import Flask, request, jsonify
from flask_cors import CORS
from config.config_path import *
import pickle
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and preprocessor
with open("model.pkl",'rb') as f:
    model = pickle.load(f)


with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data])

        processed_input = preprocessor.transform(input_df)
        prediction = model.predict(processed_input)[0]

        # Convert to int and map to label
        prediction = int(prediction)
        label_map = {
            0: 'Deceased',
            1: 'Recovered',
            2: 'Under Treatment'
        }
        result_label = label_map.get(prediction, 'Unknown')

        return jsonify({'prediction': result_label})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
