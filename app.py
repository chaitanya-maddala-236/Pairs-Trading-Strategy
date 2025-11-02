import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.stattools import coint
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Pairs Trading Strategy",
    layout="wide",
    page_icon="⚖️"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        max-width: 1600px;
        margin: 0 auto;
        padding: 2rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: white !important;
        font-size: 0.9rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border: none;
        padding: 0.8rem;
        border-radius: 10px;
        font-size: 1rem;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    .io-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("⚖️ Pairs Trading Strategy")
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #667eea;'>Statistical Arbitrage & Mean Reversion</p>", unsafe_allow_html=True)
st.markdown("---")

# Main layout with sidebar
col_sidebar, col_main = st.columns([1, 3])

with col_sidebar:
    st.markdown("### ⚙️ Strategy Controls")
    
    st.markdown("#### 📊 Market Data")
    sector = st.selectbox(
        "Sector Focus",
        ["Technology", "Financial", "Energy", "Healthcare", "Consumer"],
        index=0
    )
    
    lookback_years = st.slider("Lookback Period (years)", 1, 5, 2)
    
    st.markdown("#### 📈 Statistical Calculations")
    min_cointegration = st.slider("Min Cointegration P-value", 0.01, 0.10, 0.05, 0.01,
                                  help="Hedge ratio, mean, std")
    
    st.markdown("#### 🎯 Thresholds")
    entry_zscore = st.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.1,
                             help="Long/Short entries")
    exit_zscore = st.slider("Exit Z-Score", 0.0, 1.0, 0.5, 0.1)
    stop_loss = st.slider("Stop Loss Z-Score", 2.5, 5.0, 3.5, 0.1)
    
    st.markdown("#### 💰 Capital")
    capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, 10000,
                              help="Investment amount")
    max_positions = st.slider("Max Concurrent Pairs", 1, 10, 3)
    
    st.markdown("#### 📊 Strategy Results")
    st.markdown("""
    <div class='io-box'>
        <b>Outputs:</b>
        <ul style='font-size: 0.85rem; margin: 5px 0;'>
            <li>PnL, cumulative return</li>
            <li>Spread/z-score charts</li>
            <li>Trade signals</li>
            <li>Returns, Sharpe ratio</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    run_strategy = st.button("🚀 RUN STRATEGY", type="primary")

# Sector stock mappings
SECTOR_STOCKS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CSCO"],
    "Financial": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK", "ABT", "TMO"],
    "Consumer": ["AMZN", "HD", "NKE", "MCD", "SBUX", "TGT", "LOW", "WMT"]
}

@st.cache_data(ttl=3600, show_spinner=False)
def download_data(tickers, start, end):
    """Download stock data with robust error handling"""
    all_data = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=end)
            
            if not hist.empty and len(hist) > 50:
                if 'Close' in hist.columns:
                    all_data[ticker] = hist['Close']
        except:
            continue
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data)
    df = df.dropna()
    
    return df if len(df) > 50 else None

def find_cointegrated_pairs(data, significance=0.05):
    """Find cointegrated pairs using Engle-Granger test"""
    n = data.shape[1]
    pairs = []
    pvalue_matrix = np.ones((n, n))
    
    columns = data.columns.tolist()
    
    for i in range(n):
        for j in range(i+1, n):
            try:
                stock1 = data.iloc[:, i].values
                stock2 = data.iloc[:, j].values
                
                result = coint(stock1, stock2)
                pvalue = result[1]
                pvalue_matrix[i, j] = pvalue
                
                if pvalue < significance:
                    correlation = np.corrcoef(stock1, stock2)[0, 1]
                    
                    pairs.append({
                        'stock1': columns[i],
                        'stock2': columns[j],
                        'pvalue': pvalue,
                        'correlation': correlation
                    })
            except:
                continue
    
    return pairs, pvalue_matrix

def calculate_spread(stock1_prices, stock2_prices):
    """Calculate spread using linear regression"""
    stock1 = stock1_prices.values if hasattr(stock1_prices, 'values') else stock1_prices
    stock2 = stock2_prices.values if hasattr(stock2_prices, 'values') else stock2_prices
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(stock2, stock1)
    spread = stock1 - (slope * stock2 + intercept)
    
    spread_mean = np.mean(spread)
    spread_std = np.std(spread)
    zscore = (spread - spread_mean) / spread_std
    
    return spread, zscore, slope, intercept

def backtest_pair(data, stock1, stock2, entry_z, exit_z, stop_z):
    """Backtest a single pair"""
    stock1_prices = data[stock1]
    stock2_prices = data[stock2]
    
    spread, zscore, hedge_ratio, intercept = calculate_spread(stock1_prices, stock2_prices)
    
    zscore_series = pd.Series(zscore, index=data.index)
    spread_series = pd.Series(spread, index=data.index)
    
    trades = []
    
    in_position = False
    entry_date = None
    entry_zscore_val = None
    position_type = None
    
    for i in range(1, len(zscore)):
        date = data.index[i]
        z = zscore[i]
        
        if not in_position:
            if z > entry_z:
                in_position = True
                entry_date = date
                entry_zscore_val = z
                position_type = 'short_spread'
            elif z < -entry_z:
                in_position = True
                entry_date = date
                entry_zscore_val = z
                position_type = 'long_spread'
        else:
            should_exit = False
            exit_reason = None
            
            if position_type == 'short_spread':
                if z < exit_z:
                    should_exit = True
                    exit_reason = 'mean_reversion'
                elif z > stop_z:
                    should_exit = True
                    exit_reason = 'stop_loss'
            else:
                if z > -exit_z:
                    should_exit = True
                    exit_reason = 'mean_reversion'
                elif z < -stop_z:
                    should_exit = True
                    exit_reason = 'stop_loss'
            
            if should_exit:
                if position_type == 'short_spread':
                    pnl = (entry_zscore_val - z) * np.std(spread)
                else:
                    pnl = (z - entry_zscore_val) * np.std(spread)
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_zscore': entry_zscore_val,
                    'exit_zscore': z,
                    'position_type': position_type,
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                    'days_held': (date - entry_date).days
                })
                
                in_position = False
    
    return trades, zscore_series, spread_series, hedge_ratio

def calculate_strategy_metrics(trades, capital):
    """Calculate strategy performance metrics"""
    if not trades:
        return None
    
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean()
    
    total_return = (total_pnl / capital) * 100
    
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (total_trades - winning_trades) > 0 else 0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
    
    avg_holding_period = trades_df['days_held'].mean()
    
    # Calculate Sharpe Ratio
    returns = trades_df['pnl'] / capital
    sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'total_return': total_return,
        'profit_factor': profit_factor,
        'avg_holding_period': avg_holding_period,
        'winning_trades': winning_trades,
        'losing_trades': total_trades - winning_trades,
        'sharpe_ratio': sharpe_ratio
    }

# Main execution
with col_main:
    if run_strategy:
        with st.spinner('🔄 Analyzing pairs...'):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_years*365)
            
            tickers = SECTOR_STOCKS[sector]
            
            # Download data
            data = download_data(tickers, start_date, end_date)
            
            if data is None:
                st.error("❌ Unable to download data. Please try again.")
                st.stop()
            
            st.success(f"✅ Downloaded {len(data.columns)} stocks with {len(data)} days of data")
            
            # Find pairs
            pairs, pvalue_matrix = find_cointegrated_pairs(data, min_cointegration)
            
            if not pairs:
                st.warning(f"⚠️ No cointegrated pairs found. Try increasing p-value threshold.")
                st.stop()
            
            # Sort pairs by p-value
            pairs = sorted(pairs, key=lambda x: x['pvalue'])[:10]
            
            # Backtest each pair
            pair_results = []
            for pair in pairs:
                try:
                    trades, zscore, spread, hedge_ratio = backtest_pair(
                        data, pair['stock1'], pair['stock2'],
                        entry_zscore, exit_zscore, stop_loss
                    )
                    
                    metrics = calculate_strategy_metrics(trades, capital)
                    
                    if metrics and metrics['total_trades'] > 0:
                        pair_results.append({
                            'pair': pair,
                            'trades': trades,
                            'metrics': metrics,
                            'zscore': zscore,
                            'spread': spread,
                            'hedge_ratio': hedge_ratio
                        })
                except:
                    continue
            
            if not pair_results:
                st.warning("⚠️ No tradeable pairs found. Try lowering entry z-score.")
                st.stop()
            
            # Sort by total return
            pair_results = sorted(pair_results, key=lambda x: x['metrics']['total_return'], reverse=True)
        
        st.success(f"🎉 Found {len(pair_results)} tradeable pairs")
        
        # Portfolio Overview
        st.markdown("## 📊 Portfolio Overview")
        
        total_trades = sum([r['metrics']['total_trades'] for r in pair_results])
        avg_win_rate = np.mean([r['metrics']['win_rate'] for r in pair_results])
        total_return = sum([r['metrics']['total_return'] for r in pair_results[:max_positions]])
        avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in pair_results])
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Pairs", len(pair_results))
        
        with col2:
            st.metric("Total Trades", total_trades)
        
        with col3:
            st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")
        
        with col4:
            st.metric("Portfolio Return", f"{total_return:.2f}%")
        
        with col5:
            st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}")
        
        # Top pairs ranking
        st.markdown("## 🏆 Pairs Ranked by Performance")
        
        ranking_data = []
        for idx, r in enumerate(pair_results):
            ranking_data.append({
                'Rank': idx + 1,
                'Pair': f"{r['pair']['stock1']} / {r['pair']['stock2']}",
                'P-value': f"{r['pair']['pvalue']:.4f}",
                'Correlation': f"{r['pair']['correlation']:.3f}",
                'Trades': r['metrics']['total_trades'],
                'Win Rate': f"{r['metrics']['win_rate']:.1f}%",
                'Return': f"{r['metrics']['total_return']:.2f}%",
                'Sharpe': f"{r['metrics']['sharpe_ratio']:.2f}",
                'Avg Days': f"{r['metrics']['avg_holding_period']:.1f}"
            })
        
        top_pairs_df = pd.DataFrame(ranking_data)
        
        # Apply styling without causing errors
        def color_return(val):
            try:
                num = float(val.strip('%'))
                if num > 0:
                    return 'background-color: #d4edda'
                elif num < 0:
                    return 'background-color: #f8d7da'
                else:
                    return ''
            except:
                return ''
        
        st.dataframe(
            top_pairs_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Best pair analysis
        st.markdown("## 🔍 Top Pair Analysis")
        
        best_pair = pair_results[0]
        pair_info = best_pair['pair']
        
        st.info(f"**{pair_info['stock1']} / {pair_info['stock2']}** | P-value: {pair_info['pvalue']:.4f} | Correlation: {pair_info['correlation']:.3f} | Hedge Ratio: {best_pair['hedge_ratio']:.3f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Price Comparison")
            
            stock1_norm = data[pair_info['stock1']] / data[pair_info['stock1']].iloc[0] * 100
            stock2_norm = data[pair_info['stock2']] / data[pair_info['stock2']].iloc[0] * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stock1_norm.index, y=stock1_norm,
                name=pair_info['stock1'],
                line=dict(color='#667eea', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=stock2_norm.index, y=stock2_norm,
                name=pair_info['stock2'],
                line=dict(color='#764ba2', width=2)
            ))
            fig.update_layout(
                height=350,
                yaxis_title='Normalized Price',
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### ⚖️ Spread Z-Score")
            
            zscore = best_pair['zscore']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=zscore.index, y=zscore,
                fill='tozeroy',
                line=dict(color='#667eea', width=2),
                name='Z-Score'
            ))
            fig.add_hline(y=entry_zscore, line_dash="dash", line_color="red")
            fig.add_hline(y=-entry_zscore, line_dash="dash", line_color="red")
            fig.add_hline(y=exit_zscore, line_dash="dash", line_color="green")
            fig.add_hline(y=-exit_zscore, line_dash="dash", line_color="green")
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            
            fig.update_layout(
                height=350,
                yaxis_title='Z-Score',
                hovermode='x',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade history
        st.markdown("#### 📋 Trade History")
        
        trades_df = pd.DataFrame(best_pair['trades'])
        if len(trades_df) > 0:
            display_trades = trades_df[[
                'entry_date', 'exit_date', 'entry_zscore', 'exit_zscore',
                'pnl', 'days_held', 'exit_reason'
            ]].copy()
            display_trades.columns = [
                'Entry Date', 'Exit Date', 'Entry Z', 'Exit Z',
                'P&L ($)', 'Days Held', 'Exit Reason'
            ]
            display_trades['P&L ($)'] = display_trades['P&L ($)'].round(2)
            display_trades['Entry Z'] = display_trades['Entry Z'].round(2)
            display_trades['Exit Z'] = display_trades['Exit Z'].round(2)
            
            st.dataframe(display_trades, use_container_width=True, hide_index=True)
        
        # Cumulative PnL
        st.markdown("#### 💰 Cumulative P&L")
        
        if len(trades_df) > 0:
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(trades_df))),
                y=trades_df['cumulative_pnl'],
                fill='tozeroy',
                line=dict(color='#764ba2', width=3),
                name='Cumulative P&L'
            ))
            fig.update_layout(
                height=350,
                xaxis_title='Trade Number',
                yaxis_title='Cumulative P&L ($)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance Analysis
        st.markdown("## 📊 Performance Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Win/Loss")
            
            wins = best_pair['metrics']['winning_trades']
            losses = best_pair['metrics']['losing_trades']
            
            fig = go.Figure(data=[go.Pie(
                labels=['Wins', 'Losses'],
                values=[wins, losses],
                marker=dict(colors=['#2ecc71', '#e74c3c']),
                hole=.4
            )])
            fig.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Exit Reasons")
            
            if len(trades_df) > 0:
                exit_counts = trades_df['exit_reason'].value_counts()
                
                fig = go.Figure(data=[go.Bar(
                    x=exit_counts.index,
                    y=exit_counts.values,
                    marker=dict(color='#667eea')
                )])
                fig.update_layout(
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("#### P&L Distribution")
            
            if len(trades_df) > 0:
                fig = go.Figure(data=[go.Histogram(
                    x=trades_df['pnl'],
                    nbinsx=20,
                    marker=dict(color='#764ba2')
                )])
                fig.update_layout(
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Landing page
        st.markdown("### 🎯 Input/Output Flow")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='io-box'>
                <h4 style='color: #667eea; margin-top: 0;'>📥 Inputs</h4>
                <table style='width: 100%; font-size: 0.9rem;'>
                    <tr><td><b>Market Data</b></td><td>Prices of Stock A & B</td></tr>
                    <tr><td><b>Statistical Calculations</b></td><td>Hedge ratio, mean, std</td></tr>
                    <tr><td><b>Thresholds</b></td><td>Entry/exit z-score</td></tr>
                    <tr><td><b>Capital</b></td><td>Investment amount</td></tr>
                    <tr><td><b>Strategy Results</b></td><td>Lookback period</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='io-box'>
                <h4 style='color: #764ba2; margin-top: 0;'>📤 Outputs</h4>
                <table style='width: 100%; font-size: 0.9rem;'>
                    <tr><td><b>Spread chart, z-score chart</b></td></tr>
                    <tr><td><b>Trade signals</b></td></tr>
                    <tr><td><b>Long/Short entries</b></td></tr>
                    <tr><td><b>PnL, cumulative return</b></td></tr>
                    <tr><td><b>Returns, Sharpe ratio</b></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Quick Test Cases")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Test Case 1: Conservative**
            - Sector: Technology
            - Lookback: 2 years
            - Entry Z: 2.0
            - Exit Z: 0.5
            """)
        
        with col2:
            st.markdown("""
            **Test Case 2: Aggressive**
            - Sector: Financial
            - Lookback: 1 year
            - Entry Z: 1.5
            - Exit Z: 0.3
            """)
        
        with col3:
            st.markdown("""
            **Test Case 3: Long-term**
            - Sector: Healthcare
            - Lookback: 3 years
            - Entry Z: 2.5
            - Exit Z: 0.7
            """)
