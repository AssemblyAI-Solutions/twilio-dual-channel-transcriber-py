import requests

phone_number = "PHONE_NUMBER_HERE"

url = "http://localhost:8000/make_call"
params = {"phone_number": phone_number}

response = requests.post(url, params=params)
print(response.json())