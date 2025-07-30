# 🔮 PriceProphet: A Stock Direction Prediction Tool

PriceProphet is a command-line tool that uses machine learning to forecast the future direction of a stock's price. By analyzing historical data and a blend of technical indicators, it provides a data-driven signal on whether a stock is likely to move **UP** or **DOWN** over a specified period.

This project is built for learning, experimenting, and as a starting point for more complex quantitative analysis.

---

## ✨ Features

* **Powerful CLI:** Run predictions and specify parameters directly from your terminal.
* **Flexible Timeframes:** Predict price direction for the next **day, week, month,** or **year**.
* **Adjustable Prediction Bias:** Fine-tune the model's "optimism" by adjusting the `up_bias_ratio` to make it more or less likely to predict an UP signal.
* **Automated Feature Engineering:** Automatically calculates and uses a mix of indicators:
    * **Short-Term Momentum:** RSI (Relative Strength Index), MACD, and EMAs (50 & 200-day).
    * **Long-Term Momentum:** Yearly Return, Yearly Volatility, and the stock's price distance from its 200-day EMA.
* **Dynamic Data:** Fetches the most up-to-date daily stock data from Yahoo Finance (`yfinance`) every time it runs.

---

## 🚀 Getting Started

### Installation

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/Subkash2206/PriceProphet.git](https://github.com/Subkash2206/PriceProphet.git)
    cd PriceProphet
    ```

2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Create your `requirements.txt` file:**
    To ensure you have the exact libraries needed, create a file named `requirements.txt` with the following content:
    ```
    pandas
    numpy
    yfinance
    pandas-ta
    scikit-learn
    setuptools
    ```

4.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Run the tool from your terminal. The `--ticker` argument is required.

* **Predict the next day for Apple:**
    ```bash
    python predict.py --ticker AAPL
    ```
* **Predict the next week for Google:**
    ```bash
    python predict.py --ticker GOOG --duration week
    ```
* **Predict the next year for Tesla:**
    ```bash
    python predict.py --ticker TSLA --duration year
    ```



