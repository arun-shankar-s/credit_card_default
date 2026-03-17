from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load artifacts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

THRESHOLD = 0.3


# -----------------------
# FEATURE ENGINEERING
# -----------------------

def create_features(data):
    avg_bill_amt = sum(data['BILL_AMT']) / 6
    avg_pay_amt = sum(data['PAY_AMT']) / 6

    total_delay = sum(data['PAY'])

    credit_utilization = avg_bill_amt / (data['LIMIT_BAL'] + 1)
    payment_ratio = avg_pay_amt / (avg_bill_amt + 1)

    return avg_bill_amt, avg_pay_amt, total_delay, credit_utilization, payment_ratio


# -----------------------
# HOME PAGE
# -----------------------

@app.route('/')
def home():
    return render_template('index.html')


# -----------------------
# PREDICT
# -----------------------

@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get input values
        LIMIT_BAL = float(request.form['LIMIT_BAL'])
        AGE = float(request.form['AGE'])

        PAY = [float(request.form[f'PAY_{i}']) for i in [0,2,3,4,5,6]]
        BILL_AMT = [float(request.form[f'BILL_AMT{i}']) for i in range(1,7)]
        PAY_AMT = [float(request.form[f'PAY_AMT{i}']) for i in range(1,7)]

        SEX_2 = int(request.form['SEX_2'])

        EDUCATION_2 = int(request.form['EDUCATION_2'])
        EDUCATION_3 = int(request.form['EDUCATION_3'])
        EDUCATION_4 = int(request.form['EDUCATION_4'])

        MARRIAGE_2 = int(request.form['MARRIAGE_2'])
        MARRIAGE_3 = int(request.form['MARRIAGE_3'])

        # Feature engineering
        avg_bill_amt, avg_pay_amt, total_delay, credit_utilization, payment_ratio = create_features({
            "LIMIT_BAL": LIMIT_BAL,
            "PAY": PAY,
            "BILL_AMT": BILL_AMT,
            "PAY_AMT": PAY_AMT
        })

        # Build feature dictionary
        input_dict = {
            'LIMIT_BAL': LIMIT_BAL,
            'AGE': AGE,

            'PAY_0': PAY[0],
            'PAY_2': PAY[1],
            'PAY_3': PAY[2],
            'PAY_4': PAY[3],
            'PAY_5': PAY[4],
            'PAY_6': PAY[5],

            'BILL_AMT1': BILL_AMT[0],
            'BILL_AMT2': BILL_AMT[1],
            'BILL_AMT3': BILL_AMT[2],
            'BILL_AMT4': BILL_AMT[3],
            'BILL_AMT5': BILL_AMT[4],
            'BILL_AMT6': BILL_AMT[5],

            'PAY_AMT1': PAY_AMT[0],
            'PAY_AMT2': PAY_AMT[1],
            'PAY_AMT3': PAY_AMT[2],
            'PAY_AMT4': PAY_AMT[3],
            'PAY_AMT5': PAY_AMT[4],
            'PAY_AMT6': PAY_AMT[5],

            'SEX_2': SEX_2,

            'EDUCATION_2': EDUCATION_2,
            'EDUCATION_3': EDUCATION_3,
            'EDUCATION_4': EDUCATION_4,

            'MARRIAGE_2': MARRIAGE_2,
            'MARRIAGE_3': MARRIAGE_3,

            'avg_bill_amt': avg_bill_amt,
            'avg_pay_amt': avg_pay_amt,
            'total_delay': total_delay,
            'credit_utilization': credit_utilization,
            'payment_ratio': payment_ratio
        }

        # Order features correctly
        input_array = np.array([input_dict[col] for col in features]).reshape(1, -1)

        # Scale numerical features
        numerical_features = [
            'LIMIT_BAL', 'AGE',
            'avg_bill_amt', 'avg_pay_amt',
            'total_delay', 'credit_utilization',
            'payment_ratio'
        ]

        indices = [list(features).index(col) for col in numerical_features]
        input_array[:, indices] = scaler.transform(input_array[:, indices])

        # Predict
        prob = model.predict_proba(input_array)[0][1]
        prediction = "High Risk" if prob > THRESHOLD else "Low Risk"

        return render_template('index.html',
                               prediction=prediction,
                               probability=round(prob, 3))

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)