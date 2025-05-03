import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset (assumes Crop_Recommendation.csv is in the same directory)
data = pd.read_csv("Crop_Recommendation.csv")

# Define crop families and nitrogen-fixing crops for rotation logic
CROP_FAMILIES = {
    'rice': 'Poaceae', 'maize': 'Poaceae', 'wheat': 'Poaceae', 'millet': 'Poaceae',
    'chickpea': 'Fabaceae', 'lentil': 'Fabaceae', 'pigeonpeas': 'Fabaceae', 'mothbeans': 'Fabaceae',
    'mungbean': 'Fabaceae', 'blackgram': 'Fabaceae', 'kidneybeans': 'Fabaceae', 'peas': 'Fabaceae',
    'cotton': 'Malvaceae', 'coffee': 'Rubiaceae', 'jute': 'Malvaceae', 'coconut': 'Arecaceae',
    'banana': 'Musaceae', 'mango': 'Anacardiaceae', 'apple': 'Rosaceae', 'papaya': 'Caricaceae',
    'orange': 'Rutaceae', 'pomegranate': 'Lythraceae', 'grapes': 'Vitaceae', 'watermelon': 'Cucurbitaceae',
    'muskmelon': 'Cucurbitaceae'
}

NITROGEN_FIXING_CROPS = [
    'chickpea', 'lentil', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'kidneybeans', 'peas'
]

# Simple profitability scores (assumed market prices in USD per unit yield; replace with real data)
PROFITABILITY_SCORES = {
    'rice': 0.5, 'maize': 0.4, 'wheat': 0.45, 'millet': 0.35,
    'chickpea': 0.6, 'lentil': 0.65, 'pigeonpeas': 0.55, 'mothbeans': 0.5,
    'mungbean': 0.55, 'blackgram': 0.5, 'kidneybeans': 0.6, 'peas': 0.65,
    'cotton': 0.8, 'coffee': 1.2, 'jute': 0.7, 'coconut': 0.9,
    'banana': 0.7, 'mango': 1.0, 'apple': 1.1, 'papaya': 0.75,
    'orange': 0.8, 'pomegranate': 0.95, 'grapes': 1.0, 'watermelon': 0.6,
    'muskmelon': 0.65
}

# Features and target
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Encode the target labels (crop names)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training Accuracy: {train_score:.4f}")
print(f"Testing Accuracy: {test_score:.4f}")

# Save the model and label encoder
joblib.dump(model, 'crop_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Model and label encoder saved successfully.")