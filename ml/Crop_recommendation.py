#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Load the dataset
df = pd.read_excel(r"dataset.xlsx")
le = LabelEncoder()

# Step 2: Apply label encoding to 'label' and 'soil' columns
df['label'] = le.fit_transform(df['label'])
df['soil'] = le.fit_transform(df['soil'])

# Optional: Display first rows grouped by 'soil'
x = df.groupby(['soil'])
x.first()

# Step 3: Prepare the features (X) and target (Y)
X = df.drop('label', axis=1)
Y = df['label']

# Step 4: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 5: Train a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Step 6: Save the model using pickle
pickle.dump(model, open("model.pkl", "wb"))

# Step 7: Test the model accuracy
print(f"Model accuracy: {model.score(X_test, y_test)}")

# Step 8: Take user input for prediction
def get_input():
    try:
        n = int(input("Enter Nitrogen content: "))
        p = int(input("Enter Phosphorous content: "))
        k = int(input("Enter Potassium content: "))
        temp = float(input("Enter temperature (°C): "))
        humid = float(input("Enter humidity (%): "))
        ph = float(input("Enter soil pH: "))
        rain = float(input("Enter rainfall (mm): "))

        print('0. Alluvial Soil\n1. Black Soil\n2. Forest Soil\n3. Laterite Soil\n4. Red Soil')
        soil = int(input("Enter soil type (0-4): "))

        return [[n, p, k, temp, humid, ph, rain, soil]]

    except ValueError:
        print("Invalid input. Please enter numeric values only.")
        return None

# Get the input and make a prediction
user_input = get_input()
if user_input:
    crop_prediction = model.predict(user_input)
    crops = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'cotton', 'grapes', 'jute',
             'lentil', 'maize', 'mango', 'muskmelon', 'orange', 'papaya', 'pigeonpeas',
             'pomegranate', 'rice', 'watermelon']

    predicted_crop = crops[crop_prediction[0]]
    print(f"Recommended crop: {predicted_crop}")
