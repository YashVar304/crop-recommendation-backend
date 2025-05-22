import numpy as np
import pickle

# Load the model and scaler
randclf = pickle.load(open('model.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Crop dictionary to map numbers back to crop names
crop_dict_reverse = {
    1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans', 5: 'pigeonpeas', 6: 'mothbeans', 7: 'mungbean',
    8: 'blackgram', 9: 'lentil', 10: 'pomegranate', 11: 'banana', 12: 'mango', 13: 'grapes', 14: 'watermelon',
    15: 'muskmelon', 16: 'apple', 17: 'orange', 18: 'papaya', 19: 'coconut', 20: 'cotton', 21: 'jute', 22: 'coffee'
}

# Function to take user input
def user_input():
    print("Enter the input parameters for crop recommendation:")
    N = int(input("Nitrogen (N) [0-140]: "))
    P = int(input("Phosphorus (P) [5-145]: "))
    K = int(input("Potassium (K) [5-205]: "))
    temperature = float(input("Temperature (°C) [8.0-43.0]: "))
    humidity = float(input("Humidity (%) [14.0-99.0]: "))
    ph = float(input("pH Level [3.5-9.5]: "))
    rainfall = float(input("Rainfall (mm) [20.0-300.0]: "))
    
    data = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    return data

# Function to recommend crop
def recommend_crop(input_data):
    features = np.array([[input_data['N'], input_data['P'], input_data['K'],
                           input_data['temperature'], input_data['humidity'],
                           input_data['ph'], input_data['rainfall']]])
    
    scaled_features = mx.transform(features)
    prediction = randclf.predict(scaled_features)[0]
    
    # Map the number back to the crop name
    predicted_crop = crop_dict_reverse[prediction]
    return predicted_crop

# Main program flow
if __name__ == "__main__":
    print("Welcome to the Crop Recommendation System 🌾")
    print("This program predicts the best crop to grow based on input parameters.")
    
    input_data = user_input()
    
    print("\nInput Parameters:")
    for key, value in input_data.items():
        print(f"  {key}: {value}")
    
    result = recommend_crop(input_data)
    
    print(f"\nRecommended Crop: {result.capitalize()}")
    print("\nThank you for using the Crop Recommendation System!")
