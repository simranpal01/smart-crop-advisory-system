# ========================================================
# 🌾 Smart Crop Recommendation System (Flask + Animation)
# ========================================================
from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__, static_folder="static")

# Load model and scaler
with open("crop_recommendation_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

        return render_template('index.html', prediction_text=f"🌱 Recommended Crop: {prediction.capitalize()}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# ✅ Corrected here
if __name__ == "__main__":
    app.run(debug=True)
