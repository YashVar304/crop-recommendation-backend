from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and scaler
randclf = pickle.load(open('model.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Crop dictionary
crop_dict_reverse = {
    1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans', 5: 'pigeonpeas', 6: 'mothbeans', 7: 'mungbean',
    8: 'blackgram', 9: 'lentil', 10: 'pomegranate', 11: 'banana', 12: 'mango', 13: 'grapes', 14: 'watermelon',
    15: 'muskmelon', 16: 'apple', 17: 'orange', 18: 'papaya', 19: 'coconut', 20: 'cotton', 21: 'jute', 22: 'coffee'
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['N'], data['P'], data['K'], 
                          data['temperature'], data['humidity'], 
                          data['ph'], data['rainfall']]])
    
    scaled_features = mx.transform(features)
    prediction = randclf.predict(scaled_features)[0]
    predicted_crop = crop_dict_reverse[prediction]

    return jsonify({'crop': predicted_crop})

if __name__ == '__main__':
    app.run(debug=True)
