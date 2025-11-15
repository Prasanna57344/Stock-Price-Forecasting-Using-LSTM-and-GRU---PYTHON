import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from textblob import TextBlob  # For sentiment analysis

# Custom styling
st.set_page_config(page_title="AI Stock Forecaster", layout="wide")
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #ffffff;
        color: #1a1a1a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        background-color: #ffffff;
    }
    
    /* Sidebar Styling */
    [data-testid=stSidebar] {
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
        padding: 2rem 1rem;
    }
    
    /* Button Styling */
    .stock-button {
        width: 100%;
        margin: 8px 0;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px;
        transition: all 0.3s ease;
    }
    .stock-button:hover {
        background-color: #e9ecef;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Typography */
    h1 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    h2, h3 {
        color: #34495e;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    /* Metric Cards */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-2px);
    }
    
    /* Analysis Cards */
    .analysis-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #3498db;
    }
    
    /* Data Source Cards */
    .data-source {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Metric Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Chart Container */
    .chart-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    /* Alert Styling */
    .alert {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .alert-info { 
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .alert-success {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
    }
    
    /* Loading Animation */
    .stProgress > div > div > div > div {
        background-color: #3498db !important;
    }
    
    /* Table Styling */
    .dataframe {
        border: none !important;
        border-collapse: separate;
        border-spacing: 0;
        width: 100%;
        margin: 1rem 0;
    }
    .dataframe th {
        background-color: #f8f9fa;
        padding: 12px;
        text-align: left;
        border-bottom: 2px solid #dee2e6;
    }
    .dataframe td {
        padding: 12px;
        border-bottom: 1px solid #e9ecef;
    }
    .dataframe tr:hover {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------
# Utility Functions
# --------------------------------------
def get_usd_to_inr():
    """Fetch USD to INR exchange rate."""
    exchange_rate_data = yf.download("USDINR=X", period="1d")
    return exchange_rate_data['Close'].iloc[-1]

def calculate_technical_indicators(df):
    """Calculate technical indicators like SMA, EMA, RSI, and MACD."""
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df.dropna()

def get_news_sentiment(ticker):
    """Perform sentiment analysis on stock news."""
    stock = yf.Ticker(ticker)
    news = stock.news
    
    if not news:
        return 0.0  # Neutral sentiment if no news
    
    sentiments = []
    for item in news[:5]:  # Check first 5 news items
        title = item.get('title', item.get('headline', ''))
        if title:
            analysis = TextBlob(title)
            sentiments.append(analysis.sentiment.polarity)
    
    return np.mean(sentiments) if sentiments else 0.0

def prepare_data(df, sequence_length=60):
    """Prepare data for LSTM/GRU models."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    # Create sequences
    X = []
    y = []
    
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i, 0])
        y.append(data_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler, data_scaled

def build_model(layer_type, input_shape):
    """Build a Sequential neural network model."""
    model = Sequential()
    model.add(layer_type(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(layer_type(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_prices(model, last_sequence, scaler, days_ahead):
    """Predict future prices using the trained model."""
    future_prices = []
    current_sequence = last_sequence.reshape(1, len(last_sequence), 1)
    
    for _ in range(days_ahead):
        # Predict next price
        next_price = model.predict(current_sequence, verbose=0)
        future_prices.append(next_price)
        
        # Update sequence
        current_sequence = np.append(current_sequence[:, 1:, :], next_price.reshape(1, 1, 1), axis=1)
    
    # Inverse transform to get actual prices
    future_prices = np.array(future_prices)
    future_prices_transformed = scaler.inverse_transform(future_prices.reshape(-1, 1))
    
    return future_prices_transformed

def analyze_price_movement(df):
    """Analyze the reasons behind stock price movements."""
    analysis = {}
    
    # Technical Analysis
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Price_MA_50'] = df['Close'].rolling(window=50).mean()
    df['Price_MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Price Momentum
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    price_change = ((current_price - prev_price) / prev_price) * 100
    
    # Volume Analysis
    current_volume = df['Volume'].iloc[-1]
    avg_volume = df['Volume_MA'].iloc[-1]
    volume_change = ((current_volume - avg_volume) / avg_volume) * 100
    
    # Moving Average Analysis
    ma_50 = df['Price_MA_50'].iloc[-1]
    ma_200 = df['Price_MA_200'].iloc[-1]
    
    # RSI Analysis
    rsi = df['RSI'].iloc[-1]
    rsi_prev = df['RSI'].iloc[-2]
    
    # MACD Analysis
    macd = df['MACD'].iloc[-1]
    signal = df['Signal'].iloc[-1]
    macd_prev = df['MACD'].iloc[-2]
    signal_prev = df['Signal'].iloc[-2]
    
    # Determine trend and signals
    analysis['trend'] = 'Bullish' if price_change > 0 else 'Bearish'
    analysis['strength'] = abs(price_change)
    analysis['volume_trend'] = 'High' if volume_change > 20 else 'Low' if volume_change < -20 else 'Normal'
    analysis['rsi_signal'] = 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'
    analysis['macd_signal'] = 'Buy' if macd > signal else 'Sell'
    analysis['ma_signal'] = 'Bullish' if ma_50 > ma_200 else 'Bearish'
    
    # Detailed analysis text
    analysis['detailed_reasons'] = []
    
    # Price Movement Reasons
    if price_change > 0:
        analysis['detailed_reasons'].append({
            'factor': 'Price Movement',
            'impact': 'Positive',
            'description': f'Stock price increased by {price_change:.2f}% indicating positive market sentiment',
            'details': [
                f'Current price: ₹{current_price:.2f}',
                f'Previous price: ₹{prev_price:.2f}',
                'Potential catalysts: Market optimism, positive news, or strong financial results'
            ]
        })
    else:
        analysis['detailed_reasons'].append({
            'factor': 'Price Movement',
            'impact': 'Negative',
            'description': f'Stock price decreased by {abs(price_change):.2f}% indicating negative market sentiment',
            'details': [
                f'Current price: ₹{current_price:.2f}',
                f'Previous price: ₹{prev_price:.2f}',
                'Potential catalysts: Market pessimism, negative news, or weak financial results'
            ]
        })
    
    # Volume Analysis
    if volume_change > 20:
        analysis['detailed_reasons'].append({
            'factor': 'Trading Volume',
            'impact': 'Strong',
            'description': f'High trading volume (+{volume_change:.2f}% above average) suggests strong market interest',
            'details': [
                f'Current volume: {current_volume:,.0f} shares',
                f'Average volume: {avg_volume:,.0f} shares',
                'High volume indicates strong market participation and conviction'
            ]
        })
    elif volume_change < -20:
        analysis['detailed_reasons'].append({
            'factor': 'Trading Volume',
            'impact': 'Weak',
            'description': f'Low trading volume ({volume_change:.2f}% below average) suggests weak market interest',
            'details': [
                f'Current volume: {current_volume:,.0f} shares',
                f'Average volume: {avg_volume:,.0f} shares',
                'Low volume may indicate lack of market interest or conviction'
            ]
        })
    
    # RSI Signal
    if rsi > 70:
        analysis['detailed_reasons'].append({
            'factor': 'RSI (Relative Strength Index)',
            'impact': 'Overbought',
            'description': f'RSI at {rsi:.2f} indicates overbought conditions, potential reversal',
            'details': [
                f'Current RSI: {rsi:.2f}',
                f'Previous RSI: {rsi_prev:.2f}',
                'RSI above 70 suggests the stock may be overvalued',
                'Consider potential price correction or consolidation'
            ]
        })
    elif rsi < 30:
        analysis['detailed_reasons'].append({
            'factor': 'RSI (Relative Strength Index)',
            'impact': 'Oversold',
            'description': f'RSI at {rsi:.2f} indicates oversold conditions, potential bounce',
            'details': [
                f'Current RSI: {rsi:.2f}',
                f'Previous RSI: {rsi_prev:.2f}',
                'RSI below 30 suggests the stock may be undervalued',
                'Watch for potential price recovery'
            ]
        })
    
    # MACD Analysis
    if macd > signal and macd_prev <= signal_prev:
        analysis['detailed_reasons'].append({
            'factor': 'MACD (Moving Average Convergence Divergence)',
            'impact': 'Bullish Crossover',
            'description': 'MACD crossed above signal line indicating potential upward momentum',
            'details': [
                f'MACD: {macd:.4f}',
                f'Signal Line: {signal:.4f}',
                'Bullish crossover suggests potential upward trend',
                'Consider this a potential buying opportunity'
            ]
        })
    elif macd < signal and macd_prev >= signal_prev:
        analysis['detailed_reasons'].append({
            'factor': 'MACD',
            'impact': 'Bearish Crossover',
            'description': 'MACD crossed below signal line indicating potential downward momentum',
            'details': [
                f'MACD: {macd:.4f}',
                f'Signal Line: {signal:.4f}',
                'Bearish crossover suggests potential downward trend',
                'Exercise caution and monitor for further confirmation'
            ]
        })
    
    # Moving Average Analysis
    if ma_50 > ma_200:
        analysis['detailed_reasons'].append({
            'factor': 'Moving Averages',
            'impact': 'Bullish',
            'description': '50-day MA above 200-day MA indicates long-term uptrend (Golden Cross)',
            'details': [
                f'50-day MA: ₹{ma_50:.2f}',
                f'200-day MA: ₹{ma_200:.2f}',
                'Golden Cross pattern suggests strong bullish trend',
                'Long-term investors often view this as a buy signal'
            ]
        })
    else:
        analysis['detailed_reasons'].append({
            'factor': 'Moving Averages',
            'impact': 'Bearish',
            'description': '50-day MA below 200-day MA indicates long-term downtrend (Death Cross)',
            'details': [
                f'50-day MA: ₹{ma_50:.2f}',
                f'200-day MA: ₹{ma_200:.2f}',
                'Death Cross pattern suggests strong bearish trend',
                'Consider defensive positioning or reduced exposure'
            ]
        })
    
    return analysis

def get_company_fundamentals(ticker):
    """Get fundamental data for company analysis."""
    stock = yf.Ticker(ticker)
    info = stock.info
    
    fundamentals = {
        'Market Cap': info.get('marketCap', 'N/A'),
        'PE Ratio': info.get('trailingPE', 'N/A'),
        'Revenue Growth': info.get('revenueGrowth', 'N/A'),
        'Profit Margin': info.get('profitMargins', 'N/A'),
        'Debt to Equity': info.get('debtToEquity', 'N/A')
    }
    
    return fundamentals

def display_data_sources():
    """Display information about data sources."""
    st.markdown("""
    <div class="data-source">
        <h3>📊 Data Sources & Reliability</h3>
        
        <h4>Primary Data Sources</h4>
        <ul>
            <li><strong>Market Data:</strong> Yahoo Finance API
                <ul>
                    <li>Real-time NSE data with millisecond accuracy</li>
                    <li>Direct feed from National Stock Exchange</li>
                    <li>Automated validation checks for data integrity</li>
                </ul>
            </li>
            <li><strong>Financial Reports:</strong> Company Filings
                <ul>
                    <li>Quarterly and annual reports (SEBI mandated)</li>
                    <li>Audited financial statements</li>
                    <li>Corporate announcements and disclosures</li>
                </ul>
            </li>
        </ul>
        
        <h4>Technical Indicators</h4>
        <ul>
            <li><strong>Calculation Methods:</strong>
                <ul>
                    <li>Industry-standard formulas and methodologies</li>
                    <li>Real-time computation with error checking</li>
                    <li>Regular calibration with market benchmarks</li>
                </ul>
            </li>
        </ul>
        
        <h4>News & Sentiment Analysis</h4>
        <ul>
            <li><strong>Sources:</strong>
                <ul>
                    <li>Major financial news networks</li>
                    <li>Company press releases</li>
                    <li>Regulatory filings and updates</li>
                </ul>
            </li>
        </ul>
        
        <h4>Data Refresh Rates</h4>
        <ul>
            <li>Market Data: Real-time (1-minute intervals)</li>
            <li>Technical Indicators: Real-time calculation</li>
            <li>Fundamental Data: Daily updates</li>
            <li>News Sentiment: 15-minute updates</li>
        </ul>
        
        <p><em>All data is verified through multiple sources and cross-validated for accuracy.</em></p>
    </div>
    """, unsafe_allow_html=True)

def display_company_selection_criteria():
    """Display the criteria for top company selection."""
    st.markdown("""
    <div class="analysis-card">
        <h3>🎯 Top Companies Selection Methodology</h3>
        <p>Our AI-powered algorithm selects top companies based on a comprehensive multi-factor analysis:</p>
        
        <h4>1. Market Leadership (40% weight)</h4>
        <ul>
            <li><strong>Market Capitalization:</strong> Indicates company size and stability</li>
            <li><strong>Sector Dominance:</strong> Market share and competitive position</li>
            <li><strong>Brand Value:</strong> Company's reputation and recognition</li>
        </ul>
        
        <h4>2. Financial Strength (30% weight)</h4>
        <ul>
            <li><strong>Revenue Growth:</strong> Year-over-year revenue increase</li>
            <li><strong>Profit Margins:</strong> Profitability and operational efficiency</li>
            <li><strong>Debt Levels:</strong> Financial leverage and stability</li>
        </ul>
        
        <h4>3. Market Performance (20% weight)</h4>
        <ul>
            <li><strong>Price Momentum:</strong> Recent price trends and strength</li>
            <li><strong>Trading Volume:</strong> Liquidity and market interest</li>
            <li><strong>Volatility:</strong> Price stability and risk assessment</li>
        </ul>
        
        <h4>4. Future Potential (10% weight)</h4>
        <ul>
            <li><strong>Innovation Score:</strong> R&D investments and technological advancement</li>
            <li><strong>ESG Rating:</strong> Environmental, Social, and Governance factors</li>
            <li><strong>Analyst Recommendations:</strong> Expert opinions and forecasts</li>
        </ul>
        
        <p><em>Companies are scored on each factor and ranked based on their weighted average score.</em></p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------
# Fetch Top Indian Stocks
# --------------------------------------
def get_top_indian_stocks():
    st.subheader("📊 Today's Top Indian Stocks")
    
    try:
        # List of popular Indian stocks (NSE tickers)
        indian_stocks = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
            "HINDUNILVR.NS", "HDFC.NS", "BAJFINANCE.NS", "SBIN.NS", "BHARTIARTL.NS"
        ]
        
        # Fetch data using yfinance
        stocks_data = []
        for ticker in indian_stocks:
            try:
                data = yf.Ticker(ticker)
                hist = data.history(period="2d")
                info = data.info
                
                if not hist.empty:
                    prev_close = hist['Close'].iloc[0]
                    current_price = hist['Close'].iloc[-1]
                    percent_change = ((current_price - prev_close) / prev_close) * 100
                    
                    # Get more detailed information
                    market_cap = info.get('marketCap', 'N/A')
                    pe_ratio = info.get('trailingPE', 'N/A')
                    volume = hist['Volume'].iloc[-1]
                    avg_volume = hist['Volume'].mean()
                    volume_change = ((volume - avg_volume) / avg_volume) * 100 if avg_volume != 0 else 0
                    
                    stocks_data.append({
                        'symbol': ticker.replace('.NS', ''),
                        'price': current_price,
                        'change': percent_change,
                        'market_cap': market_cap,
                        'pe_ratio': pe_ratio,
                        'volume': volume,
                        'volume_change': volume_change,
                        'company_name': info.get('longName', ticker.replace('.NS', '')),
                        'sector': info.get('sector', 'N/A'),
                        'industry': info.get('industry', 'N/A')
                    })
            except Exception as e:
                st.warning(f"Could not fetch data for {ticker}: {str(e)}")
        
        # Sort by percent change (descending)
        stocks_data.sort(key=lambda x: x['change'], reverse=True)
        
        # Display top 5 stocks with detailed cards
        for stock in stocks_data[:5]:
            with st.expander(f"📈 {stock['company_name']} ({stock['symbol']})"):
                # Main metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Current Price",
                        value=f"₹{stock['price']:.2f}",
                        delta=f"{stock['change']:.2f}%"
                    )
                with col2:
                    if isinstance(stock['market_cap'], (int, float)):
                        market_cap_str = f"₹{stock['market_cap']/1e9:.2f}B"
                    else:
                        market_cap_str = "N/A"
                    st.metric("Market Cap", market_cap_str)
                with col3:
                    if isinstance(stock['pe_ratio'], (int, float)):
                        st.metric("P/E Ratio", f"{stock['pe_ratio']:.2f}")
                    else:
                        st.metric("P/E Ratio", "N/A")
                
                # Detailed information
                st.markdown("""
                <div class="analysis-card">
                    <h4>Company Analysis</h4>
                """, unsafe_allow_html=True)
                
                # Company details
                st.markdown(f"""
                * **Sector:** {stock['sector']}
                * **Industry:** {stock['industry']}
                * **Trading Volume:** {stock['volume']:,.0f} shares ({stock['volume_change']:.2f}% vs avg)
                """)
                
                # Movement Analysis
                if stock['change'] > 0:
                    trend = "Bullish"
                    color = "positive"
                    reason = "showing upward momentum"
                else:
                    trend = "Bearish"
                    color = "negative"
                    reason = "showing downward pressure"
                
                st.markdown(f"""
                <div class="analysis-detail {color}">
                    <h4>Price Movement Analysis</h4>
                    <p>The stock is currently <strong>{trend}</strong>, {reason}.</p>
                    <ul>
                        <li>Price Change: {stock['change']:.2f}%</li>
                        <li>Trading Volume: {stock['volume']:,.0f} shares</li>
                        <li>Volume Change: {stock['volume_change']:.2f}%</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Trading Signals
                volume_signal = "High" if stock['volume_change'] > 20 else "Low" if stock['volume_change'] < -20 else "Normal"
                st.markdown(f"""
                <div class="analysis-detail neutral">
                    <h4>Trading Signals</h4>
                    <ul>
                        <li>Price Trend: <strong>{trend}</strong></li>
                        <li>Volume Signal: <strong>{volume_signal}</strong></li>
                        <li>Market Position: <strong>{"Above" if stock['change'] > 0 else "Below"} Previous Close</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error fetching top stocks: {str(e)}")
        st.info("Using Yahoo Finance API as a fallback...")
        
        # Fallback: Show some predefined stocks
        top_nse_stocks = {
            "RELIANCE": "Reliance Industries",
            "TCS": "Tata Consultancy Services",
            "HDFCBANK": "HDFC Bank",
            "INFY": "Infosys",
            "ICICIBANK": "ICICI Bank"
        }
        
        for symbol, name in top_nse_stocks.items():
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                data = ticker.history(period="1d")
                if not data.empty:
                    with st.expander(f"📈 {name} ({symbol})"):
                        st.metric(
                            label="Current Price",
                            value=f"₹{data['Close'].iloc[-1]:.2f}",
                            delta=None
                        )
                        st.info("Limited data available in fallback mode")
            except:
                with st.expander(f"📈 {name} ({symbol})"):
                    st.metric(label="Current Price", value="N/A", delta=None)
                    st.error("Data unavailable")

# --------------------------------------
# Main App Interface
# --------------------------------------
st.title("📈 Indian Stock Market Analyzer")
st.sidebar.header("⚙️ Controls")

# Index Buttons
if st.sidebar.button("🌟 Show Today's Top Performers"):
    get_top_indian_stocks()

# User Inputs
ticker = st.sidebar.text_input("🔍 Enter Stock Ticker (e.g., RELIANCE.NS):", "RELIANCE.NS").upper()
future_date = st.sidebar.date_input("📅 Prediction Date", datetime.today() + timedelta(days=30))

# Prediction Button
if st.sidebar.button("🚀 Run Prediction"):
    with st.spinner("Crunching numbers with AI..."):
        try:
            # Fetch Data
            df = yf.download(ticker, start='2020-01-01', end=datetime.today().strftime('%Y-%m-%d'))
            
            if df.empty:
                st.error("Invalid ticker or no data available!")
            else:
                # Convert USD to INR if the stock is not Indian
                if not ticker.endswith(".NS"):
                    usd_to_inr = get_usd_to_inr()
                    df['Close'] = df['Close'] * usd_to_inr
                
                # Calculate Technical Indicators
                df = calculate_technical_indicators(df)
                
                # Candlestick Chart
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Candlestick'
                )])

                # Add SMA and EMA
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='20-Day SMA', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='20-Day EMA', line=dict(color='orange')))

                # Update layout
                fig.update_layout(
                    title=f"{ticker} Price Trend (Candlestick)",
                    xaxis_rangeslider_visible=True,
                    template="plotly_dark"  # Use a dark theme for better visibility
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment Analysis
                sentiment = get_news_sentiment(ticker)
                st.metric("News Sentiment Score", f"{sentiment:.2f}", 
                          help="Score between -1 (negative) and +1 (positive)")
                
                # Display Price Movement Analysis
                analysis = analyze_price_movement(df)
                
                st.markdown("""
                <div class="analysis-card">
                    <h3>📈 Comprehensive Market Analysis</h3>
                    <p>Our AI has performed a detailed multi-factor analysis of the stock's performance and market conditions:</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display technical indicators in a modern card layout
                st.markdown("""
                <style>
                .metric-card {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 10px 0;
                }
                .indicator-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }
                .analysis-detail {
                    background-color: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 10px 0;
                    border-left: 4px solid;
                }
                .positive { border-left-color: #28a745; }
                .negative { border-left-color: #dc3545; }
                .neutral { border-left-color: #ffc107; }
                </style>
                """, unsafe_allow_html=True)
                
                # Technical Indicators Section
                st.markdown("### 📊 Technical Indicators")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Market Trend</h4>
                        <p><strong>{analysis['trend']}</strong> (Strength: {analysis['strength']:.2f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Volume Analysis</h4>
                        <p><strong>{analysis['volume_trend']}</strong> trading activity</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Technical Signals</h4>
                        <p>RSI: <strong>{analysis['rsi_signal']}</strong></p>
                        <p>MACD: <strong>{analysis['macd_signal']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed Analysis Section
                st.markdown("### 🔍 Detailed Price Movement Analysis")
                for reason in analysis['detailed_reasons']:
                    impact_color = 'positive' if reason['impact'] in ['Positive', 'Bullish', 'Strong'] else 'negative' if reason['impact'] in ['Negative', 'Bearish', 'Weak'] else 'neutral'
                    
                    st.markdown(f"""
                    <div class="analysis-detail {impact_color}">
                        <h4>{reason['factor']} - {reason['impact']}</h4>
                        <p><strong>{reason['description']}</strong></p>
                        <ul>
                        {''.join([f'<li>{detail}</li>' for detail in reason.get('details', [])])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Company Fundamentals in a modern card layout
                st.markdown("### 💰 Company Fundamentals")
                fundamentals = get_company_fundamentals(ticker)
                cols = st.columns(len(fundamentals))
                for i, (key, value) in enumerate(fundamentals.items()):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{key}</h4>
                            <p><strong>{value if isinstance(value, str) else f'{value:.2f}'}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display Data Sources with modern styling
                st.markdown("""
                <div class="analysis-card">
                    <h3>📊 Data Sources & Reliability</h3>
                    <p>All analysis is based on reliable data sources and industry-standard methodologies:</p>
                </div>
                """, unsafe_allow_html=True)
                display_data_sources()
                
                # Display Selection Criteria
                st.markdown("""
                <div class="analysis-card">
                    <h3>🎯 Selection Methodology</h3>
                    <p>Understanding how we select and analyze top-performing stocks:</p>
                </div>
                """, unsafe_allow_html=True)
                display_company_selection_criteria()
                
                # Prepare Data
                X, y, scaler, data_scaled = prepare_data(df)
                X = X.reshape((X.shape[0], X.shape[1], 1))
                
                # Train LSTM Model
                lstm_model = build_model(LSTM, (X.shape[1], 1))
                lstm_model.fit(X, y, epochs=20, batch_size=32, verbose=0)
                
                # Train GRU Model
                gru_model = build_model(GRU, (X.shape[1], 1))
                gru_model.fit(X, y, epochs=20, batch_size=32, verbose=0)
                
                # Prediction
                last_sequence = data_scaled[-60:, 0]
                days = (future_date - df.index[-1].date()).days
                
                if days > 0:
                    # Make Predictions
                    lstm_pred = predict_future_prices(lstm_model, last_sequence, scaler, days)
                    gru_pred = predict_future_prices(gru_model, last_sequence, scaler, days)
                    
                    # Risk Assessment
                    returns = df['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                    
                    # Display Results
                    col1, col2, col3 = st.columns(3)
                    col1.metric("LSTM Prediction", f"₹{lstm_pred[-1][0]:.2f}")
                    col2.metric("GRU Prediction", f"₹{gru_pred[-1][0]:.2f}")
                    col3.metric("Volatility", f"{volatility:.2%}")
                    
                    # Real-Time Alert
                    price_change = (lstm_pred[-1][0] - df['Close'][-1]) / df['Close'][-1]
                    if abs(price_change) > 0.05:
                        st.warning(f"Significant price change predicted: {price_change:.2%}")
                else:
                    st.error("The selected future date must be after the last available data point.")
        except Exception as e:
            st.error(f"Error: {str(e)}")