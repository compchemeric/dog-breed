import requests

# ✅ Change this if your server runs on a different port
url = "http://localhost:8000/predict"

# ✅ Replace this with the path to a real dog image on your system
image_path = "black-labrador.jpg"

with open(image_path, "rb") as image_file:
    files = {"file": (image_path, image_file, "image/jpeg")}
    response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response:", response.json())