from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np

app = Flask(__name__)
CORS(app)

# Load models and scaler
scaler = joblib.load('scaler.pkl')
models = joblib.load('models.pkl')
with open('model_results.json') as f:
    meta = json.load(f)

FEATURES = meta['features']
SCALED_MODELS = ['Logistic Regression', 'K-Nearest Neighbors', 'SVM', 'Deep Learning (MLP)']

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        X = np.array([[data[f] for f in FEATURES]])
        X_scaled = scaler.transform(X)

        results = {}
        for name, model in models.items():
            inp = X_scaled if name in SCALED_MODELS else X
            prob = model.predict_proba(inp)[0][1]
            results[name] = round(float(prob) * 100, 1)

        avg = round(sum(results.values()) / len(results), 1)
        return jsonify({'predictions': results, 'ensemble': avg, 'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/meta', methods=['GET'])
def get_meta():
    return jsonify({
        'accuracies': meta['accuracies'],
        'features': FEATURES,
        'importances': meta['importances'],
        'dl_info': meta.get('dl_info', {})
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models_loaded': list(models.keys())})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
