import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# STEP 1: CREATE MOCK DATA IF NO FILE EXISTS
# ==========================================
# This ensures your script runs smoothly even if a file is missing
if not os.path.exists("car_data.csv"):
    print("car_data.csv not found! Generating a sample dataset...")
    np.random.seed(42)
    sample_data = {
        "Year": np.random.randint(2012, 2024, 100),
        "Kilometers_Driven": np.random.randint(10000, 120000, 100),
        "Fuel_Type": np.random.choice(["Petrol", "Diesel", "CNG"], 100),
        "Transmission": np.random.choice(["Manual", "Automatic"], 100),
        "Selling_Price": np.random.uniform(2.0, 15.0, 100) # Price in Lakhs
    }
    df = pd.DataFrame(sample_data)
    df.to_csv("car_data.csv", index=False)
else:
    df = pd.read_csv("car_data.csv")

print("Dataset Loaded. Shape:", df.shape)

# ==========================================
# STEP 2: DATA PREPROCESSING
# ==========================================
# Drop missing rows
df = df.dropna()

# Convert Year to Car Age (Current Year: 2026)
if "Year" in df.columns:
    df["Car_Age"] = 2026 - df["Year"]
    df = df.drop(columns=["Year"])

# Convert text labels into numbers (One-Hot Encoding)
categorical_cols = [col for col in ["Fuel_Type", "Transmission"] if col in df.columns]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Processed DataFrame Columns:", df.columns.tolist())

# ==========================================
# STEP 3: TRAIN TEST SPLIT
# ==========================================
# Target column is assumed to be 'Selling_Price'
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# STEP 4: MODEL TRAINING & EVALUATION
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} Lakhs")
print(f"R-squared (R2) Score: {r2:.4f} ({r2*100:.2f}% variance explained)")

# ==========================================
# STEP 5: SAVE MODEL FILE
# ==========================================
with open("car_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as 'car_price_model.pkl'!")
