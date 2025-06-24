import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, flash
import joblib

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev')

# Load model, scaler, and metrics at startup
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
METRICS_PATH = "model_metrics.txt"

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        return None, None

trained_model, scaler = load_model()

# Load metrics if available
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return f.read()
    return None
model_metrics = load_metrics()

def make_prediction(form):
    if trained_model is None or scaler is None:
        print("Model or scaler not loaded.")
        return None, None
    try:
        # Extract and validate form data
        fields = [
            ('perc', float),
            ('age', int),
            ('income', int),
            ('count1', float),
            ('count2', float),
            ('count3', float),
            ('aus', float),
            ('nop', int),
            ('s_channel', str),
            ('r_area', str)
        ]
        data = []
        for name, typ in fields:
            val = form.get(name, '')
            if val == '':
                print(f"Missing value for {name}")
                raise ValueError(f"Missing value for {name}")
            if typ == str:
                data.append(val)
            else:
                data.append(typ(val))
        # One-hot encode categorical features
        s_channels = ['A', 'B', 'C', 'D', 'E']
        r_areas = ['Urban', 'Rural']
        s_channel_vec = [1 if data[8] == ch else 0 for ch in s_channels]
        r_area_vec = [1 if data[9] == area else 0 for area in r_areas]
        features = data[:8] + s_channel_vec + r_area_vec
        print(f"Features for prediction: {features}")
        # Scale features
        X = scaler.transform([features])
        # Predict
        pred = trained_model.predict(X)[0]
        prob = trained_model.predict_proba(X)[0][1]
        print(f"Prediction: {pred}, Probability: {prob}")
        return pred, prob
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

@app.route("/", methods=['GET', 'POST'])
def home():
    # Only show the form and metrics, no error messages on GET
    return render_template('index.html', metrics=model_metrics)

@app.route("/predict", methods=["POST"])
def result():
    pred, prob = make_prediction(request.form)
    if pred is None or prob is None:
        flash("Invalid input or model error. Please check your entries.")
        print("Returned error to user on /predict.")
        return render_template('index.html', metrics=model_metrics)
    if pred == 1:
        msg = f"Will Pay Next Premium (Confidence: {prob:.2%})"
    else:
        msg = f"Will Default Next Premium (Confidence: {(1-prob):.2%})"
    return render_template('predict.html', prediction=msg)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)