from flask import Flask, request, jsonify, render_template
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
        # Get JSON data from the POST request
        data = request.json
        print(data)  # Debug: log the received data
        
        # Convert the data to the appropriate format
        features = np.array([[
            data['s4'], data['s7'], data['s8'], data['s9'], data['s11'],
            data['s12'], data['s13'], data['s14'], data['av2'], data['av3'],
            data['av4'], data['av7'], data['av8'], data['av9'], data['av11'],
            data['av12'], data['av13'], data['av14'], data['av15'], data['av17'],
            data['av20'], data['av21']
        ]], dtype=float)

        # Standardize the features
        features = scaler.transform(features)

        # Predict using the model
        y_pred_continuous = model.predict(features)

        # Convert continuous predictions to binary using a threshold (e.g., 0.5)
        threshold = 0.5
        y_pred = (y_pred_continuous > threshold).astype(int)

        # Determine if the engine is suitable for use
        decision = "Layak digunakan" if y_pred[0] == 1 else "Tidak layak digunakan"

        return jsonify({
            'predicted_label': int(y_pred[0]),
            'decision': decision
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
