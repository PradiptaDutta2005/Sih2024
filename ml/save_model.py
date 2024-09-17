import pickle
from sklearn.ensemble import RandomForestClassifier # type: ignore
import numpy as np

# Example data
X_train = np.array([[1, 2, 3, 25, 50, 6.5, 200, 0], [4, 5, 6, 30, 60, 6.0, 300, 1]])
y_train = np.array([0, 1])  # Example target values

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully!")
