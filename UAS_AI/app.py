from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('linear_regression_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    scaler = joblib.load('scaler.pkl')
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the feature values from the form
        features = [float(request.form[feature]) for feature in ['feature1', 'feature2']]
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        prediction_continuous = model.predict(features)
        prediction = (prediction_continuous > 0.5).astype(int)
        if prediction[0] == 1:
            prediction_text = 'Aircraft is fit to fly (Layak dipakai).'
        else:
            prediction_text = 'Aircraft is not fit to fly (Tidak layak dipakai).'
        return render_template('index.html', prediction_text=prediction_text)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
