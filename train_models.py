import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load the dataset
df = pd.read_csv('data/co2_emission.csv')

# Prepare data for SLR (Engine Size only)
X_slr = df[['Engine Size']]
y = df['CO2 Emission']

# Prepare data for MLR (Engine Size, Cylinders, Fuel Type)
X_mlr = df[['Engine Size', 'Cylinders', 'Fuel Type']]

# Split data for SLR
X_slr_train, X_slr_test, y_slr_train, y_slr_test = train_test_split(
    X_slr, y, test_size=0.2, random_state=42
)

# Split data for MLR
X_mlr_train, X_mlr_test, y_mlr_train, y_mlr_test = train_test_split(
    X_mlr, y, test_size=0.2, random_state=42
)

# Train SLR model
slr_model = LinearRegression()
slr_model.fit(X_slr_train, y_slr_train)

# Train MLR model
mlr_model = LinearRegression()
mlr_model.fit(X_mlr_train, y_mlr_train)

# Evaluate models
slr_pred = slr_model.predict(X_slr_test)
mlr_pred = mlr_model.predict(X_mlr_test)

slr_r2 = r2_score(y_slr_test, slr_pred)
mlr_r2 = r2_score(y_mlr_test, mlr_pred)

print(f"SLR R² Score: {slr_r2:.4f}")
print(f"MLR R² Score: {mlr_r2:.4f}")

# Save models
with open('models/slr_model.pkl', 'wb') as f:
    pickle.dump(slr_model, f)

with open('models/mlr_model.pkl', 'wb') as f:
    pickle.dump(mlr_model, f)

print("Models saved successfully!") 