# Indian Used Car Price Prediction

This project uses **Multiple Linear Regression** to predict the resale value of used cars in India based on key vehicle attributes.

## 🛠️ Features Used
* **Car_Age**: Calculated from the manufacturing year.
* **Kilometers_Driven**: Total distance the car has traveled.
* **Fuel_Type**: Petrol, Diesel, or CNG (One-Hot Encoded).
* **Transmission**: Manual or Automatic (One-Hot Encoded).

## 🚀 Getting Started
1. Clone this repository to your local computer.
2. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script to train the model:
   ```bash
   python model.py
   ```

## 📊 Results
* **R² Score:** Explains the variance in resale prices.
* **Mean Absolute Error (MAE):** Average prediction deviation in INR.
