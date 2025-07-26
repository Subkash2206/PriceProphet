# PriceProphet

PriceProphet

PriceProphet is a stock price direction prediction tool currently using a Random Forest Classifier trained on historical stock data and engineered features. It predicts whether a stock’s price will go up or down for a short timeframe (currently fixed to 1 day, with plans for more).
About PriceProphet

PriceProphet aims to provide a data-driven signal on the directional movement of stocks using machine learning. The project is in an early development stage — the model currently predicts one-day price direction with moderate accuracy (~53%) using basic features like moving averages and lagged returns. This is a proof-of-concept and not financial advice.
Features (Current)

  - Predicts short-term (1 day) stock price movement direction.

  - Uses Random Forest Classifier on engineered features including:

  - 50-day and 200-day EMAs

  - Lagged returns

  - EMA differences

  - Data fetched via Yahoo Finance (yfinance).

  - Basic CLI interface (future argparse support planned).

Installation

Clone this repo:

```bash
git clone git@github.com:Subkash2206/PriceProphet.git
cd PriceProphet
```

(Optional) Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
    pip install -r requirements.txt
```

This predicts whether the stock price will rise or fall the next day.

How It Works

   - Downloads historical stock data from Yahoo Finance.

   - Extracts features like EMAs and returns.

   - Labels data with the next day’s price movement direction (up/down).

   - Trains a Random Forest Classifier on this data.

   - Predicts direction for the next day.

Data Sources

  - Historical price and volume data via the yfinance Python library.

Model Details

   - Algorithm: Random Forest Classifier

   - Features: EMAs, lagged returns, EMA differences

   - Training: Supervised learning on labeled historical data

  - Current accuracy: ~53% (slightly better than random guessing)

Limitations & Disclaimer

  - The model is a basic prototype and should not be used for real financial decisions.

  - Accuracy is modest; predictions are probabilistic signals, not guarantees.

  - Market conditions and external factors are not modeled.

  - Always do your own research before acting on predictions.

Roadmap

  - Add argparse for flexible CLI input (ticker, timeframe, model choice)

  - Extend prediction horizon beyond 1 day

  - Experiment with more advanced models (XGBoost, LSTM)

  - Add backtesting and evaluation tools

  - Build a user-friendly interface/dashboard


Contributions and suggestions are welcome. Feel free to fork and open pull requests.
License

This project is licensed under the MIT License. See the LICENSE file for details.
Contact

Created by Subhash Kashyap



