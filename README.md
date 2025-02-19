

# Stock Market Analysis & Prediction

## Overview
This project analyzes stock market trends and predicts stock prices using **Yahoo Finance API** and **machine learning**. The model utilizes historical stock data and applies **Linear Regression** to forecast future prices.

## Dataset
- Data Source: **Yahoo Finance API**
- Features:
  - Open price
  - High price
  - Low price
  - Close price
  - Trading volume
  - Moving averages (50-day & 200-day)
  
## Technologies Used
- **Python**
- **Jupyter Notebook**
- **yfinance** (for fetching stock data)
- **Pandas & NumPy** (for data manipulation)
- **Matplotlib & Seaborn** (for visualization)
- **Scikit-Learn** (for machine learning)

## Implementation Steps
1. **Fetch Stock Data**
   - Use `yfinance` to retrieve historical stock prices
   - Visualize trends with Matplotlib
   
2. **Feature Engineering**
   - Calculate moving averages (50-day & 200-day)
   - Handle missing values

3. **Model Training**
   - Prepare input features (`Close`, `50_MA`, `200_MA`)
   - Split dataset into training and testing sets
   - Train a **Linear Regression** model

4. **Prediction & Evaluation**
   - Predict stock prices for test data
   - Evaluate model performance using **Mean Absolute Error (MAE)**
   - Visualize actual vs. predicted prices

## How to Run the Project
1. Install dependencies:
   ```sh
   pip install yfinance pandas numpy matplotlib seaborn scikit-learn
   ```
2. Run the Jupyter Notebook:
   ```sh
   jupyter notebook Stock_Market_Analysis.ipynb
   ```
3. Execute each cell to fetch data, train the model, and make predictions.

## Results
- Model Performance: [Add evaluation metrics here]
- Insights: [Summarize key findings]

## Future Enhancements
- Improve accuracy using **LSTMs (Long Short-Term Memory)**
- Incorporate additional features like **RSI, MACD, Bollinger Bands**
- Deploy as a **Flask or Streamlit web app**


