import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (ensure the CSV file has 'Price' and 'Sales' columns)
file_path = "C:/Users/USer/Documents/Mind Melters/price_sales_data.csv" # Update with your dataset path
data = pd.read_csv(file_path)

# Display basic statistics and first few rows
print(data.describe())
print(data.head())

# Visualize the relationship between Price and Sales
plt.scatter(data['Price'], data['Sales'], color='blue')
plt.title('Price vs Sales')
plt.xlabel('Price')
plt.ylabel('Sales')
plt.show()

# Prepare data for regression
X = data[['Price']].values  # Independent variable (Price)
y = data['Sales'].values    # Dependent variable (Sales)

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Model parameters
print(f"Coefficient (Slope): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Predict sales using the model
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the regression line
plt.scatter(data['Price'], data['Sales'], color='blue', label='Actual Sales')
plt.plot(data['Price'], y_pred, color='red', label='Predicted Sales')
plt.title('Price vs Sales with Regression Line')
plt.xlabel('Price')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Find the optimal price
prices = np.linspace(data['Price'].min(), data['Price'].max(), 100).reshape(-1, 1)
sales_predictions = model.predict(prices)

optimal_price = prices[np.argmax(sales_predictions)][0]
print(f"Optimal Price for Maximum Sales: {optimal_price}")
