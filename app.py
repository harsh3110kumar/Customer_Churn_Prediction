from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Load model artifacts
with open('churn_model.pkl', 'rb') as f:
    dt = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define feature columns (updated to only include required columns)
categorical_cols = ['Gender']
numerical_cols = [
    'Age', 'Support Calls', 'Payment Delay', 
    'Total Spend', 'Last Interaction'
]

def get_retention_recommendation(customer_data, prediction):
    """
    Provides retention recommendations based on customer profile and churn prediction.
    """
    if prediction == 0:
        return {
            "action": "No immediate action needed",
            "message": "Customer is not predicted to churn. Maintain current engagement.",
            "recommendations": []
        }

    recommendations = []

    # Support Calls-based recommendations
    if customer_data['Support Calls'] >= 8:
        if customer_data['Support Calls'] >= 15:
            recommendations.append({
                "feature": "VIP Support",
                "description": "Immediate escalation to senior support team with 24hr resolution SLA",
                "rationale": f"Extremely high support calls ({customer_data['Support Calls']}) indicate serious unresolved issues"
            })
        else:
            recommendations.append({
                "feature": "Dedicated Support",
                "description": "Assign a dedicated account manager for immediate issue resolution",
                "rationale": f"Multiple support calls ({customer_data['Support Calls']}) suggest recurring problems"
            })

    # Payment Delay-based recommendations
    if customer_data['Payment Delay'] > 10:
        if customer_data['Payment Delay'] > 30:
            recommendations.append({
                "feature": "Payment Relief",
                "description": "Offer payment plan with first month free and reduced installments",
                "rationale": f"Severe payment delay ({customer_data['Payment Delay']} days) indicates financial distress"
            })
        else:
            recommendations.append({
                "feature": "Payment Flexibility",
                "description": "Waive late fees and extend due date by 2 weeks",
                "rationale": f"Payment delay ({customer_data['Payment Delay']} days) may indicate temporary cash flow issues"
            })

    # Total Spend-based recommendations
    if customer_data['Total Spend'] < 1000:
        recommendations.append({
            "feature": "Value Boost",
            "description": "Free upgrade to premium features for 60 days",
            "rationale": f"Mid-range spending (${customer_data['Total Spend']}) suggests opportunity to demonstrate value"
        })
    else:
        recommendations.append({
            "feature": "Elite Retention",
            "description": "Personalized account review with executive team and custom benefits package",
            "rationale": f"High-value customer (${customer_data['Total Spend']}) worth exceptional retention efforts"
        })

    # Age-based recommendations
    if customer_data['Age'] <= 44:
        recommendations.append({
            "feature": "Next-Gen Engagement",
            "description": "Access to beta features and innovation community",
            "rationale": f"Younger customer (age {customer_data['Age']}) may value cutting-edge features"
        })

    # Last Interaction-based recommendations
    if customer_data['Last Interaction'] > 20:
        recommendations.append({
            "feature": "Reactivation Campaign",
            "description": "Personalized 'We want you back' offer with time-sensitive benefits",
            "rationale": f"{customer_data['Last Interaction']} days since last interaction indicates disengagement"
        })

    # Prioritize recommendations
    prioritized_recommendations = sorted(
        recommendations,
        key=lambda x: 1 if "VIP" in x["feature"] else
                      2 if "Payment" in x["feature"] else 3
    )

    return {
        "action": "Immediate retention action required",
        "message": f"Customer matches {len(recommendations)} key churn indicators",
        "recommendations": prioritized_recommendations
    }

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/predict.html', methods=['GET'])
def serve_predict():
    return render_template('predict.html')

@app.route('/feedback.html', methods=['GET'])
def serve_feedback():
    return render_template('feedback.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        logging.info(f"Received input data: {data}")

        # Validate all fields present
        for col in categorical_cols + numerical_cols:
            if col not in data:
                return jsonify({'error': f"Missing field: {col}"}), 400

        # Encode categorical features
        encoded_data = data.copy()
        for col in categorical_cols:
            encoded_data[col] = label_encoders[col].transform([encoded_data[col]])[0]

        # Build DataFrame & scale numeric features
        df = pd.DataFrame([encoded_data])
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        df = df[dt.feature_names_in_]

        # Predict with probabilities
        pred = dt.predict(df)[0]
        proba = dt.predict_proba(df)[0][1]
        
        # Get recommendations
        recommendations = get_retention_recommendation(data, pred)
        
        # Identify key risk factors
        risk_factors = []
        if data['Support Calls'] >= 8:
            risk_factors.append(f"High support calls ({data['Support Calls']})")
        if data['Payment Delay'] > 10:
            risk_factors.append(f"Payment delay ({data['Payment Delay']} days)")
        if data['Last Interaction'] > 20:
            risk_factors.append(f"Recent inactivity ({data['Last Interaction']} days)")

        return jsonify({
            'prediction': 'Yes' if pred == 1 else 'No',
            'probability': f"{proba:.1%}",
            'risk_factors': risk_factors,
            'recommendations': recommendations
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)