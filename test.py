import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "The acting was ok!"}
response = requests.post(url, json=data)

print(response.json())