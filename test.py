import requests

url = "http://127.0.0.1:5001/predict"

# Low risk
# data = {
#     "num_delays": 0,
#     "avg_util": 0.188246,
#     "avg_pay_ratio": 0.615141,
#     "credit_stress": 0.0166571
# }

# High risk
data = {
    "num_delays": 3,
    "avg_util": 0.82,
    "avg_pay_ratio": 0.21,
    "credit_stress": 0.65
}

res = requests.post(url, json=data)

print("Status Code:", res.status_code)

if res.status_code == 200:
    print("Response:", res.json())
else:
    print("Error Response:", res.text)  