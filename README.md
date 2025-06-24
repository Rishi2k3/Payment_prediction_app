# Payment Prediction App

A modern, robust, and production-ready web app to predict whether a customer will pay their next insurance premium, built with Flask and scikit-learn.

## Features
- **Machine Learning Model:** Predicts premium payment using a RandomForest with class balancing and SMOTE.
- **Modern UI:** Beautiful, responsive, and user-friendly green-themed interface.
- **Robust Error Handling:** User-friendly messages and backend logging.
- **Easy Deployment:** Ready for Heroku or local deployment.
- **Input Validation:** All fields have clear ranges and types.

## Tech Stack
- Python 3.8+
- Flask
- scikit-learn
- imbalanced-learn
- pandas, numpy, joblib
- HTML5, CSS3 (custom, responsive, modern)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Predicting-if-a-customer-will-default-their-next-premium-main
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
If you want to retrain the model:
```bash
python3 train_model.py
```
This will generate `model.pkl` and `scaler.pkl`.

### 4. Run the App Locally
```bash
python3 app.py
```
Visit [http://localhost:5050](http://localhost:5050) in your browser.

## Deployment Guide

### Deploy to Heroku
1. **Login to Heroku:**
   ```bash
   heroku login
   ```
2. **Create a Heroku app:**
   ```bash
   heroku create your-app-name
   ```
3. **Add a Procfile (already included):**
   ```
   web: python app.py
   ```
4. **Push to Heroku:**
   ```bash
   git add .
   git commit -m "Deploy"
   git push heroku main
   ```
5. **Open your app:**
   ```bash
   heroku open
   ```

### Deploy with Docker (Optional)
- Add a `Dockerfile` if needed and build/run as usual.

## Usage
- Fill out the form with customer details.
- Click **Predict** to see if the customer will pay their next premium.
- All fields are required. Use the placeholder hints for value ranges.

## Project Structure
- `app.py` - Flask backend
- `train_model.py` - Model training script
- `model.pkl`, `scaler.pkl` - Trained model and scaler
- `templates/` - HTML templates
- `static/` - CSS styles
- `requirements.txt` - Python dependencies

## Support
If you encounter any issues, please open an issue or contact the maintainer.

---
**Enjoy your robust, modern Payment Prediction App!**
