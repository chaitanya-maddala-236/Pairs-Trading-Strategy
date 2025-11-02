# ⚖️ Pairs Trading Strategy

Statistical arbitrage system that identifies cointegrated stock pairs and executes market-neutral mean reversion trades.

## 🚀 Live Demo

**[Try it now!](https://YOUR-APP.streamlit.app)** 🔗

## 📊 Key Features

- ⚖️ **Cointegration Testing**: Engle-Granger test for pair discovery
- 📈 **Automated Backtesting**: Complete trade history with P&L
- 🎯 **Z-Score Signals**: Standardized entry/exit rules
- 💹 **Market Neutral**: Long-short hedging strategy
- 📊 **Risk Management**: Stop-loss and position sizing
- 🔥 **Heatmap Visualization**: See all pair relationships

## 🎯 Performance

- Win Rate: **XX%**
- Total Return: **XX%**
- Profit Factor: **X.XX**
- Sharpe Ratio: **X.XX**
- Market Neutral: **Beta ≈ 0**

*(Run app to see actual results)*

## 🧮 Mathematical Foundation

Uses cointegration theory to identify pairs:
```
Stock1_t = β × Stock2_t + ε_t

Where ε_t is stationary (mean-reverting)
```

### Trading Signals
```
Z-Score = (Spread - μ) / σ

Entry:  |Z| > 2.0
Exit:   |Z| < 0.5
Stop:   |Z| > 3.5
```

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run pairs_trading_app.py
```

## 📦 Tech Stack

- **Python 3.10+**
- **Streamlit** - Interactive dashboard
- **Scipy** - Statistical tests
- **Statsmodels** - Cointegration analysis
- **Plotly** - Data visualization
- **yFinance** - Market data

## 🎨 Screenshots

![Dashboard](screenshots/dashboard.png)
![Pairs Analysis](screenshots/pairs.png)
![Z-Score](screenshots/zscore.png)

## 📈 How It Works

1. **Discovery**: Test all stock pairs for cointegration
2. **Filtering**: Keep pairs with p-value < 0.05
3. **Spread**: Calculate Spread = Stock1 - β × Stock2
4. **Signals**: Generate trades based on z-score
5. **Execution**: Long undervalued, short overvalued
6. **Exit**: Close when spread mean-reverts

## 🔬 Statistical Tests

- **Engle-Granger Test**: Tests cointegration
- **ADF Test**: Tests stationarity
- **OLS Regression**: Calculates hedge ratio
- **Z-Score**: Standardizes spread

## 💼 Use Cases

- Hedge fund statistical arbitrage
- Market-neutral portfolio construction
- Risk management and hedging
- Academic research on mean reversion

## 📚 Research Basis

Based on foundational research:
- Engle & Granger (1987) - Cointegration theory
- Gatev et al. (2006) - Pairs trading performance
- Vidyamurthy (2004) - Pairs Trading methods

## 🎓 For Students/Researchers

This project demonstrates:
- ✅ Advanced statistical methods
- ✅ Time series analysis
- ✅ Market-neutral strategies
- ✅ Risk management
- ✅ Production-ready backtesting


---

⭐ If you found this helpful, please star the repo!
