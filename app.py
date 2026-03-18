from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

# -----------------------
# LOAD MODEL
# -----------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# If you used scaling during training, load it
# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# -----------------------
# FEATURE ORDER (CRITICAL)
# -----------------------
FEATURES = ['num_delays', 'avg_util', 'avg_pay_ratio', 'credit_stress']

# -----------------------
# EXPLANATION FUNCTION
# -----------------------
def explain_user(row, prob, threshold=0.5):
    explanations = []

    if prob > threshold:
        # HIGH RISK
        if row['num_delays'] > 2:
            explanations.append(f"Multiple delayed payments ({row['num_delays']})")

        if row['avg_util'] > 0.7:
            explanations.append(f"High credit utilization ({row['avg_util']:.2f})")

        if row['avg_pay_ratio'] < 0.3:
            explanations.append(f"Low repayment ratio ({row['avg_pay_ratio']:.2f})")

    else:
        # LOW RISK
        if row['num_delays'] == 0:
            explanations.append("No delayed payments")

        if row['avg_util'] < 0.3:
            explanations.append(f"Low credit utilization ({row['avg_util']:.2f})")

        if row['avg_pay_ratio'] >= 0.5:
            explanations.append(f"Good repayment ratio ({row['avg_pay_ratio']:.2f})")

    if not explanations:
        explanations.append("Moderate financial behavior")

    return explanations

# -----------------------
# ROUTES
# -----------------------
@app.route("/")
def home():
    return "Credit Risk API is running 🚀"

def get_recommendations(row):
    recs = []

    if row['avg_util'] > 0.7:
        recs.append("Reduce credit utilization below 30%")

    if row['num_delays'] > 0:
        recs.append("Avoid delayed payments")

    if row['avg_pay_ratio'] < 0.5:
        recs.append("Increase repayment ratio")

    if row['credit_stress'] > 0.5:
        recs.append("Reduce financial stress and liabilities")

    if not recs:
        recs.append("Maintain current financial behavior")

    return recs

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Validate input
        for feat in FEATURES:
            if feat not in data:
                return jsonify({"error": f"Missing feature: {feat}"}), 400

        # Maintain correct feature order
        features = np.array([data[feat] for feat in FEATURES]).reshape(1, -1)

        # Apply scaling if used
        # features = scaler.transform(features)

        # Model probability
        prob = model.predict_proba(features)[0][1]

        # Decision threshold
        threshold = 0.5
        pred = int(prob > threshold)

        # Confidence calculation
        confidence = abs(prob - threshold) * 2
        confidence_percent = round(confidence * 100, 2)

        # Risk %
        risk_percent = round(prob * 100, 2)

        # Explanation
        explanation = explain_user(data, prob, threshold)

        recommendations = get_recommendations(data)

        return jsonify({
            "prediction": "High Risk" if pred == 1 else "Low Risk",
            "metrics": {
                "default_probability": risk_percent,
                "confidence": confidence_percent
            },
            "explanation": explanation,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------
# RUN APP
# -----------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)