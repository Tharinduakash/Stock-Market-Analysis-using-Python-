import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Fetch historical stock data for a specific company (e.g., Apple)
ticker = "AAPL"
stock_data = yf.download(ticker, start="2020-01-01", end="2024-02-19")

# Display first few rows
print(stock_data.head())

# Plot closing price
plt.figure(figsize=(10,5))
plt.plot(stock_data['Close'], label="Closing Price")
plt.title(f"{ticker} Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()


# Create moving average features
stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()
stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()

# Drop NaN values
stock_data = stock_data.dropna()

# Plot moving averages
plt.figure(figsize=(10,5))
plt.plot(stock_data['Close'], label="Closing Price")
plt.plot(stock_data['50_MA'], label="50-Day MA", linestyle='dashed')
plt.plot(stock_data['200_MA'], label="200-Day MA", linestyle='dashed')
plt.legend()
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Prepare features (X) and target variable (y)
stock_data['Tomorrow'] = stock_data['Close'].shift(-1)
X = stock_data[['Close', '50_MA', '200_MA']][:-1]
y = stock_data['Tomorrow'].dropna()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Compare actual vs. predicted prices
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Prices")
plt.plot(y_pred, label="Predicted Prices", linestyle='dashed')
plt.title("Stock Price Prediction")
plt.legend()
plt.show()
