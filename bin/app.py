from flask import Flask, render_template, request, jsonify
from linear_regression import load_model, data, get_regression_params
import numpy as np

Model_File = 'results/model.pkl'

model = load_model(Model_File)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    if not input_data or 'features' not in input_data:
        return jsonify({'error':'no features provided'}), 400

    features = input_data['features']
    if not isinstance(features, list) or len(features) != 10:
        return jsonify({'error':'Features should be a list of ten numbers'}), 400

    features_array = np.array(features).reshape(1,-1)
    prediction = model.predict(features_array)
    return jsonify({'prediction': prediction[0]})

@app.route('/regression-line', methods=['GET'])
def regression_line():
    params = get_regression_params(model)
    return jsonify(params)

@app.route('/dataset', methods=['GET'])
def dataset():
    X_list = data.data.tolist()
    y_list = data.target.tolist()
    return jsonify({'features': X_list, 'target': y_list})

if __name__ == '__main__':
    app.run(debug=True)
