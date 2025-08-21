import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

# Simulate dataset
np.random.seed(42)
n = 500
temperature = np.random.normal(70, 10, n)  # avg 70°C
vibration = np.random.normal(5, 2, n)      # avg 5 mm/s

# Define "failure" when temperature > 85 or vibration > 8
labels = ((temperature > 85) | (vibration > 8)).astype(int)

# Create dataframe
df = pd.DataFrame({
    "time": range(n),
    "temperature": temperature,
    "vibration": vibration,
    "label": labels
})

# Save dataset
df.to_csv("sensor_data.csv", index=False)

# Train model
X = df[["temperature", "vibration"]]
y = df["label"]

model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Save model
joblib.dump(model, "pdm_model.pkl")

print("✅ Model and dataset created successfully!")
