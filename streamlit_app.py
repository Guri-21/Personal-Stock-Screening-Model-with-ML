import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🚀 AI Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stock-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and return stock data"""
    try:
        df = pd.read_csv('stock_data.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Stock data not found! Please run the Jupyter notebook first to collect data.")
        st.stop()

@st.cache_data
def prepare_ml_data(df):
    """Prepare data for ML model (same as notebook)"""
    ml_df = df.copy()
    
    # Fill missing values with median for numeric columns
    numeric_cols = ['pe_ratio', 'debt_to_equity', 'roe', 'profit_margin', 
                   'revenue_growth', 'beta', 'peg_ratio', 'forward_pe']
    
    for col in numeric_cols:
        if col in ml_df.columns:
            ml_df[col] = ml_df[col].fillna(ml_df[col].median())
    
    # Create investment score (same logic as notebook)
    required_cols = ['annual_return', 'volatility', 'pe_ratio']
    for col in required_cols:
        if col not in ml_df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()
    
    # Create scores
    returns_clean = ml_df['annual_return'].replace([np.inf, -np.inf], np.nan).fillna(0)
    returns_score = (returns_clean - returns_clean.min()) / (returns_clean.max() - returns_clean.min()) if returns_clean.max() != returns_clean.min() else pd.Series([0.5] * len(ml_df))
    
    volatility_clean = ml_df['volatility'].replace([np.inf, -np.inf], np.nan).fillna(ml_df['volatility'].median())
    volatility_score = 1 - ((volatility_clean - volatility_clean.min()) / (volatility_clean.max() - volatility_clean.min())) if volatility_clean.max() != volatility_clean.min() else pd.Series([0.5] * len(ml_df))
    
    pe_clean = ml_df['pe_ratio'].replace([np.inf, -np.inf], np.nan).fillna(ml_df['pe_ratio'].median())
    pe_score = 1 - ((pe_clean - pe_clean.min()) / (pe_clean.max() - pe_clean.min())) if pe_clean.max() != pe_clean.min() else pd.Series([0.5] * len(ml_df))
    
    if 'roe' in ml_df.columns and ml_df['roe'].notna().sum() > 0:
        roe_clean = ml_df['roe'].replace([np.inf, -np.inf], np.nan).fillna(0)
        roe_score = (roe_clean - roe_clean.min()) / (roe_clean.max() - roe_clean.min()) if roe_clean.max() != roe_clean.min() and roe_clean.max() > 0 else pd.Series([0.5] * len(ml_df))
    else:
        roe_score = pd.Series([0.5] * len(ml_df))
    
    ml_df['investment_score'] = 0.3 * returns_score + 0.25 * volatility_score + 0.25 * pe_score + 0.2 * roe_score
    ml_df['investment_score'] = ml_df['investment_score'].replace([np.inf, -np.inf], np.nan)
    ml_df = ml_df.dropna(subset=['investment_score'])
    
    # Select features
    feature_cols = ['pe_ratio', 'market_cap', 'annual_return', 'volatility', 'price_to_high_ratio', 'dividend_yield']
    optional_features = ['debt_to_equity', 'roe', 'profit_margin', 'beta']
    for feat in optional_features:
        if feat in ml_df.columns and ml_df[feat].notna().sum() > len(ml_df) * 0.3:
            feature_cols.append(feat)
    
    X = ml_df[feature_cols].copy()
    for col in feature_cols:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(X[col].median())
    
    y = ml_df['investment_score']
    
    if X.isna().any().any():
        X = X.fillna(0)
    if y.isna().any():
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        ml_df = ml_df[valid_idx]
    
    return X, y, ml_df, feature_cols

@st.cache_resource
def train_model(X, y):
    """Train and return ML model and scaler"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    
    return model, scaler

def screen_stocks_app(df, model, scaler, feature_cols, min_market_cap, max_pe_ratio, 
                     min_annual_return, risk_level, top_n):
    """Screen stocks for the app"""
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['market_cap'] >= min_market_cap]
    filtered_df = filtered_df[filtered_df['pe_ratio'] <= max_pe_ratio]
    filtered_df = filtered_df[filtered_df['annual_return'] >= min_annual_return]
    
    volatility_thresholds = {'Low Risk': 20, 'Medium Risk': 35, 'High Risk': 100}
    max_volatility = volatility_thresholds.get(risk_level, 35)
    filtered_df = filtered_df[filtered_df['volatility'] <= max_volatility]
    
    if len(filtered_df) == 0:
        return pd.DataFrame()
    
    X_screen = filtered_df[feature_cols].fillna(filtered_df[feature_cols].median())
    X_screen_scaled = scaler.transform(X_screen)
    predictions = model.predict(X_screen_scaled)
    
    filtered_df = filtered_df.copy()
    filtered_df['ml_score'] = predictions
    
    top_stocks = filtered_df.nlargest(top_n, 'ml_score')
    return top_stocks

# Main app
def main():
    # Header
    st.title("🚀 AI-Powered Stock Screener")
    st.markdown("### Discover the best investment opportunities using Machine Learning")
    st.markdown("---")
    
    # Load data and train model
    with st.spinner("🔄 Loading stock data and training AI model..."):
        df = load_data()
        X, y, ml_df, feature_cols = prepare_ml_data(df)
        model, scaler = train_model(X, y)
    
    # Sidebar controls
    st.sidebar.header("🎯 Screening Criteria")
    
    # Market cap filter
    market_cap_options = {
        "Large Cap (>$10B)": 10e9,
        "Mid Cap ($2B - $10B)": 2e9,
        "Small Cap ($300M - $2B)": 300e6,
        "Any Size": 0
    }
    market_cap_choice = st.sidebar.selectbox("Market Capitalization", list(market_cap_options.keys()))
    min_market_cap = market_cap_options[market_cap_choice]
    
    # PE ratio filter
    max_pe_ratio = st.sidebar.slider("Maximum P/E Ratio", 5, 50, 30)
    
    # Return filter
    min_annual_return = st.sidebar.slider("Minimum Annual Return (%)", -50, 50, -10)
    
    # Risk level
    risk_level = st.sidebar.selectbox("Risk Tolerance", ["Low Risk", "Medium Risk", "High Risk"])
    
    # Number of stocks
    top_n = st.sidebar.slider("Number of Stocks to Show", 5, 20, 10)
    
    # Screen button
    if st.sidebar.button("🔍 Screen Stocks", type="primary"):
        with st.spinner("🤖 AI is analyzing stocks..."):
            results = screen_stocks_app(ml_df, model, scaler, feature_cols, 
                                      min_market_cap, max_pe_ratio, min_annual_return, 
                                      risk_level, top_n)
        
        if len(results) == 0:
            st.warning("❌ No stocks match your criteria. Try relaxing the filters.")
        else:
            # Display results
            st.success(f"✅ Found {len(results)} stocks matching your criteria!")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average ML Score", f"{results['ml_score'].mean():.3f}")
            with col2:
                st.metric("Average Annual Return", f"{results['annual_return'].mean():.1f}%")
            with col3:
                st.metric("Average P/E Ratio", f"{results['pe_ratio'].mean():.1f}")
            with col4:
                st.metric("Average Volatility", f"{results['volatility'].mean():.1f}%")
            
            # Results table
            st.subheader("📊 Top Stock Recommendations")
            display_cols = ['ticker', 'company_name', 'current_price', 'pe_ratio', 
                          'annual_return', 'volatility', 'dividend_yield', 'ml_score', 'sector']
            
            # Format the dataframe for display
            results_display = results[display_cols].copy()
            results_display['ml_score'] = results_display['ml_score'].round(3)
            results_display['current_price'] = results_display['current_price'].round(2)
            results_display['annual_return'] = results_display['annual_return'].round(1)
            results_display['volatility'] = results_display['volatility'].round(1)
            results_display['dividend_yield'] = results_display['dividend_yield'].round(2)
            
            st.dataframe(results_display, use_container_width=True, hide_index=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 ML Score vs Annual Return")
                fig = px.scatter(results, x='ml_score', y='annual_return', 
                               hover_data=['ticker', 'company_name'],
                               title="AI Score vs Performance",
                               labels={'ml_score': 'ML Score', 'annual_return': 'Annual Return (%)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🏢 Sector Distribution")
                sector_counts = results['sector'].value_counts()
                fig = px.pie(values=sector_counts.values, names=sector_counts.index,
                           title="Recommended Stocks by Sector")
                st.plotly_chart(fig, use_container_width=True)
            
            # Download button
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="stock_screener_results.csv",
                mime="text/csv"
            )
    
    # Dataset info
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 Dataset: {len(df)} stocks analyzed\n🤖 ML Model: Random Forest\n📅 Data: Real-time from Yahoo Finance")
    
    # Footer
    st.markdown("---")
    st.markdown("### About This Project")
    st.markdown("""
    This AI-powered stock screener uses machine learning to identify promising investment opportunities:
    - **Data Source**: Real-time financial data from Yahoo Finance
    - **ML Algorithm**: Random Forest model trained on financial metrics
    - **Scoring System**: Combines returns, volatility, P/E ratio, and profitability
    - **Filtering**: Customizable criteria for different risk profiles
    """)

if __name__ == "__main__":
    main()