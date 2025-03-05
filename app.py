from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained objects
try:
    with open("churn_model.pkl", "rb") as f:
        dt = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logging.info("Pre-trained objects loaded successfully.")
except Exception as e:
    logging.error(f"Error loading pre-trained objects: {e}")
    raise

# Define categorical and numerical columns
categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
numerical_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']

# Home route to serve the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        logging.info(f"Received input data: {data}")

        # Validate input
        required_fields = categorical_cols + numerical_cols
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Encode categorical inputs
        for col in categorical_cols:
            data[col] = label_encoders[col].transform([data[col]])[0]
        logging.info("Categorical columns encoded successfully.")

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Scale numerical features
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        logging.info("Numerical features scaled successfully.")

        # Ensure column order matches training data
        input_df = input_df[dt.feature_names_in_]

        # Make prediction
        predicted_churn = dt.predict(input_df)[0]
        prediction_result = "Yes" if predicted_churn == 1 else "No"
        logging.info(f"Prediction made successfully: {prediction_result}")

        # Return prediction result
        return jsonify({"prediction": prediction_result})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)