import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm


# Load historical data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        raise ValueError(f"No data fetched for {ticker}. Check the ticker symbol or date range.")
    if 'Adj Close' not in data.columns:
        if 'Close' in data.columns:
            data['Adj Close'] = data['Close']
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' column is available in the data.")
    data['Returns'] = data['Adj Close'].pct_change()
    return data


# Define strategy (Mean Reversion + Momentum + ATR-based filtering)
def trading_strategy(data, lookback=20, rsi_period=14, atr_period=14):
    data['SMA'] = data['Adj Close'].rolling(window=lookback).mean()
    data['STD'] = data['Adj Close'].rolling(window=lookback).std()
    data['Upper_Band'] = data['SMA'] + (2 * data['STD'])
    data['Lower_Band'] = data['SMA'] - (2 * data['STD'])

    # RSI Calculation
    delta = data['Adj Close'].diff()
    gain = delta.clip(lower=0).rolling(window=rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(window=rsi_period).mean()
    data['RSI'] = 100 - (100 / (1 + gain / loss))

    # ATR Calculation
    data['High-Low'] = data['High'] - data['Low']
    data['High-Close'] = (data['High'] - data['Adj Close'].shift(1)).abs()
    data['Low-Close'] = (data['Low'] - data['Adj Close'].shift(1)).abs()
    data['TR'] = pd.concat([data['High-Low'], data['High-Close'], data['Low-Close']], axis=1).max(axis=1)
    data['ATR'] = data['TR'].rolling(window=atr_period).mean()

    # Signals: Long for mean reversion, short for momentum
    data['Signal'] = 0
    data.loc[(data['Adj Close'] < data['Lower_Band']) & (data['RSI'] < 30), 'Signal'] = 1  # Buy
    data.loc[(data['Adj Close'] > data['Upper_Band']) & (data['RSI'] > 70), 'Signal'] = -1  # Sell

    return data


# Backtesting with position sizing
def backtest(data, risk_per_trade=0.01, initial_balance=10000):
    balance = initial_balance
    position_size = []
    data['Cumulative_Strategy'] = 1

    for i in range(1, len(data)):
        if data['Signal'].iloc[i - 1] != 0:
            risk_amount = balance * risk_per_trade
            atr = data['ATR'].iloc[i - 1]
            if atr == 0 or np.isnan(atr):
                continue

            units = risk_amount / atr
            pnl = units * data['Returns'].iloc[i] * data['Adj Close'].iloc[i]
            balance += pnl
        position_size.append(balance)

    data['Cumulative_Strategy'] = pd.Series(position_size, index=data.index[-len(position_size):])
    data['Cumulative_Market'] = (1 + data['Returns']).cumprod() * initial_balance
    return data


# Monte Carlo Simulation
def monte_carlo_simulation(data, n_simulations=1000):
    daily_returns = data['Returns'].dropna()
    mean = daily_returns.mean()
    std = daily_returns.std()

    simulations = []
    for _ in range(n_simulations):
        simulated_returns = np.random.normal(mean, std, len(daily_returns))
        simulations.append((1 + simulated_returns).cumprod())

    simulations = np.array(simulations)
    return simulations


# Plotting
def plot_results(data, simulations):
    plt.figure(figsize=(14, 7))

    # Plot backtesting results
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Cumulative_Strategy'], label='Strategy', color='blue')
    plt.plot(data.index, data['Cumulative_Market'], label='Market', color='orange')
    plt.title('Strategy vs. Market Performance')
    plt.legend()

    # Plot Monte Carlo simulations
    plt.subplot(2, 1, 2)
    for sim in simulations:
        plt.plot(data.index[-len(sim):], sim, color='gray', alpha=0.1)
    plt.title('Monte Carlo Simulations')

    plt.tight_layout()
    plt.show()


# Main function
def main():
    tickers = ['^DJI', 'GC=F', 'EURUSD=X']  # US30, XAUUSD, EURUSD
    start = '2020-01-01'
    end = '2023-01-01'

    for ticker in tickers:
        print(f"Processing {ticker}...")
        data = load_data(ticker, start, end)
        data = trading_strategy(data)
        data = backtest(data)

        simulations = monte_carlo_simulation(data)

        print(f"Results for {ticker}:")
        plot_results(data, simulations)


if __name__ == "__main__":
    main()
