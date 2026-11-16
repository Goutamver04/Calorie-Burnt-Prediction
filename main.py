import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# 1. Load Data
calories = pd.read_csv('/content/calories.csv')
exercise_data = pd.read_csv('/content/exercise.csv')

# 2. Combine DataFrames
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

# 3. Preprocess Categorical Data
calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

# 4. Split Features & Target
X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
Y = calories_data['Calories']

# 5. Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# 6. Model Initialization & Training
model = XGBRegressor(objective='reg:squarederror')
model.fit(X_train, Y_train)

# 7. Prediction & Evaluation
test_data_prediction = model.predict(X_test)
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)

print(f"Data Shape: {calories_data.shape}")
print(f"Mean Absolute Error = {mae}")