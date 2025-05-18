import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample housing data
data = {
    'SquareFeet': [1500, 1800, 2400, 3000, 1200],
    'Bedrooms': [3, 4, 4, 5, 2],
    'Bathrooms': [2, 2, 3, 4, 1],
    'Price': [300000, 350000, 500000, 600000, 200000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted Prices:", y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Optional: plot predictions
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
