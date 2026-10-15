"""
CCA-2 (AI ML 1) Project
Team Members:
1. Pari Gupta - PRN: 1302250444
2. Manjiri Bankhele - PRN: 1302250623
Description: Part 1 - Traditional Programming vs Machine Learning Comparison
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =====================================================================
# 1. DATA GENERATION & ACQUISITION
# =====================================================================
np.random.seed(42)
data_size = 100
sizes = np.random.randint(800, 3500, size=data_size)
bedrooms = np.random.randint(1, 5, size=data_size)
ages = np.random.randint(0, 30, size=data_size)

# Generative pricing equation featuring target real-world noise parameters
prices = (sizes * 150) + (bedrooms * 25000) - (ages * 2000) + np.random.randint(-15000, 15000, size=data_size)

df = pd.DataFrame({
    'House_Size_SQFT': sizes,
    'Bedrooms': bedrooms,
    'Age_Years': ages,
    'Price_USD': prices
})

excel_filename = "house_data.xlsx"
df.to_excel(excel_filename, index=False)
print(f" [SUCCESS] Dataset persistent store built at: '{excel_filename}'\n")

# Load targeted frame using pandas engine
data = pd.read_excel(excel_filename)

# =====================================================================
# 2. TRADITIONAL PROGRAMMING APPROACH (Heuristic Rule-Based Logic)
# =====================================================================
def traditional_predict_price(row):
    if row['House_Size_SQFT'] < 1500:
        base = 180000
    elif 1500 <= row['House_Size_SQFT'] < 2500:
        base = 320000
    else:
        base = 500000
        
    base += row['Bedrooms'] * 15000
    # Traditional logic systematically omits asset age parameters, yielding high variance
    return base

data['Traditional_Prediction'] = data.apply(traditional_predict_price, axis=1)

# =====================================================================
# 3. MACHINE LEARNING APPROACH (Multivariate Supervised Learning)
# =====================================================================
X = data[['House_Size_SQFT', 'Bedrooms', 'Age_Years']]
y = data['Price_USD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ml_model = LinearRegression()
ml_model.fit(X_train, y_train)

data['ML_Prediction'] = ml_model.predict(X)

# Evaluation metrics extraction
ml_test_preds = ml_model.predict(X_test)
traditional_test_preds = X_test.apply(traditional_predict_price, axis=1)

# =====================================================================
# 4. QUANTITATIVE PERFORMANCE ANALYTICS
# =====================================================================
print("=================== PERFORMANCE COMPARISON MATRIX ===================")
print(f"Traditional Heuristics MAE : ${mean_absolute_error(y_test, traditional_test_preds):,.2f}")
print(f"Machine Learning Model MAE : ${mean_absolute_error(y_test, ml_test_preds):,.2f}")
print(f"Machine Learning R² Score  : {r2_score(y_test, ml_test_preds):.4f} (Goodness of Fit)")
print("=====================================================================\n")

print("====================== PREDICTION HEAD SAMPLE ======================")
preview_cols = ['House_Size_SQFT', 'Bedrooms', 'Age_Years', 'Price_USD', 'Traditional_Prediction', 'ML_Prediction']
print(data[preview_cols].head(5).to_string(index=False))
print("=====================================================================")
