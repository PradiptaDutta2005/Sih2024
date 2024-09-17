from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

print(type(model))  # Should show the type of your model, e.g., RandomForestClassifier

# Define the route for the form page
@app.route('/')
def form():
    return render_template('crop_recommendation.html')

# Define the route to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from request
        data = request.form
        nitrogen = float(data['nitrogen'])
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        soil = data['soil']

        # Encode soil types (based on your model encoding)
        soil_types = {
            'Alluvial Soil': 0,
            'Black Soil': 1,
            'Forest': 2,
            'Laterite': 3,
            'Red Soil': 4
        }
        soil_encoded = soil_types.get(soil, 0)

        # Create the input array for the model
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, soil_encoded]])

        # Predict the crop using the model
        prediction = model.predict(input_data)[0]

        # List of crops corresponding to the model's output
        crops = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'cotton', 'grapes', 'jute',
                 'lentil', 'maize', 'mango', 'muskmelon', 'orange', 'papaya', 'pigeonpeas',
                 'pomegranate', 'rice', 'watermelon']
        crop_name = crops[prediction]

        # Send the result back to the frontend
        return jsonify({'crop': crop_name})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
