import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Sample disease dataset loading
# Replace this with actual dataset path

data = {
    'age': [25, 30, 45, 35, 50, 23],
    'bp': [120, 130, 140, 135, 150, 125],
    'cholesterol': [220, 240, 250, 230, 260, 210],
    'disease': [0, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop('disease', axis=1)
y = df['disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {acc}")

# Save the model to a file
with open('disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Predict function

def predict_disease(input_data):
    """Predict disease based on input features"""
    with open('disease_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    pred = loaded_model.predict([input_data])
    return pred[0]

if __name__ == '__main__':
    # Example input [age, bp, cholesterol]
    sample_input = [40, 130, 240]
    result = predict_disease(sample_input)
    print(f"Predicted Disease (0=no, 1=yes): {result}")
