import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Page configuration
st.set_page_config(
    page_title="NYZTrade Halal Stock Screener",
    page_icon="☪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Islamic green theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    h1, h2, h3 {
        color: #10b981 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin: 10px 0;
    }
    .halal-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    .non-halal-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    .doubtful-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    .compliance-card {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .non-compliance-card {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #10b981;
    }
    .stProgress > div > div {
        background-color: #10b981;
    }
</style>
""", unsafe_allow_html=True)


# Shariah Compliance Criteria - AAOIFI Standards and Common Islamic Finance Principles
SHARIAH_STANDARDS = {
    "AAOIFI": {
        "name": "Accounting and Auditing Organization for Islamic Financial Institutions",
        "debt_to_market_cap": 0.33,  # Total debt / Market cap <= 33%
        "interest_income_ratio": 0.05,  # Interest income / Total revenue <= 5%
        "liquid_assets_ratio": 0.33,  # (Cash + Interest-bearing securities) / Market cap <= 33%
        "non_compliant_income_ratio": 0.05,  # Non-compliant income / Total revenue <= 5%
    },
    "DJIM": {
        "name": "Dow Jones Islamic Market Index",
        "debt_to_market_cap": 0.33,
        "interest_income_ratio": 0.05,
        "liquid_assets_ratio": 0.33,
        "non_compliant_income_ratio": 0.05,
    },
    "S&P": {
        "name": "S&P Shariah Indices",
        "debt_to_market_cap": 0.33,
        "interest_income_ratio": 0.05,
        "liquid_assets_ratio": 0.33,
        "non_compliant_income_ratio": 0.05,
    }
}

# Non-Halal Business Activities
NON_HALAL_SECTORS = [
    "Alcohol",
    "Tobacco",
    "Gambling",
    "Pork Production",
    "Conventional Banking",
    "Conventional Insurance",
    "Weapons & Defense",
    "Adult Entertainment",
    "Interest-Based Financial Services"
]

# Keywords for non-halal business detection
NON_HALAL_KEYWORDS = [
    "alcohol", "liquor", "beer", "wine", "brewery", "distillery",
    "tobacco", "cigarette", "cigar",
    "casino", "gambling", "lottery", "betting",
    "pork", "pig", "swine",
    "bank", "banking", "finance", "insurance", "nbfc",
    "defense", "weapons", "arms", "military",
    "adult", "entertainment"
]

# Industry benchmarks (same as previous)
INDUSTRY_BENCHMARKS = {
    "Technology": {"pe": 28, "ev_ebitda": 18},
    "Financial Services": {"pe": 15, "ev_ebitda": 10},
    "Consumer Cyclical": {"pe": 20, "ev_ebitda": 12},
    "Healthcare": {"pe": 25, "ev_ebitda": 15},
    "Industrials": {"pe": 18, "ev_ebitda": 11},
    "Energy": {"pe": 12, "ev_ebitda": 8},
    "Basic Materials": {"pe": 14, "ev_ebitda": 9},
    "Consumer Defensive": {"pe": 22, "ev_ebitda": 13},
    "Communication Services": {"pe": 16, "ev_ebitda": 10},
    "Real Estate": {"pe": 20, "ev_ebitda": 14},
    "Utilities": {"pe": 16, "ev_ebitda": 9},
    "Default": {"pe": 20, "ev_ebitda": 12}
}

MIDCAP_BENCHMARKS = {
    "Technology": {"pe": 25, "ev_ebitda": 16},
    "Financial Services": {"pe": 14, "ev_ebitda": 9},
    "Consumer Cyclical": {"pe": 18, "ev_ebitda": 11},
    "Healthcare": {"pe": 22, "ev_ebitda": 14},
    "Industrials": {"pe": 16, "ev_ebitda": 10},
    "Energy": {"pe": 11, "ev_ebitda": 7},
    "Basic Materials": {"pe": 13, "ev_ebitda": 8},
    "Consumer Defensive": {"pe": 20, "ev_ebitda": 12},
    "Communication Services": {"pe": 15, "ev_ebitda": 9},
    "Real Estate": {"pe": 18, "ev_ebitda": 13},
    "Utilities": {"pe": 15, "ev_ebitda": 8},
    "Default": {"pe": 18, "ev_ebitda": 11}
}

SMALLCAP_BENCHMARKS = {
    "Technology": {"pe": 22, "ev_ebitda": 14},
    "Financial Services": {"pe": 12, "ev_ebitda": 8},
    "Consumer Cyclical": {"pe": 16, "ev_ebitda": 10},
    "Healthcare": {"pe": 20, "ev_ebitda": 12},
    "Industrials": {"pe": 15, "ev_ebitda": 9},
    "Energy": {"pe": 10, "ev_ebitda": 6},
    "Basic Materials": {"pe": 12, "ev_ebitda": 7},
    "Consumer Defensive": {"pe": 18, "ev_ebitda": 11},
    "Communication Services": {"pe": 14, "ev_ebitda": 8},
    "Real Estate": {"pe": 16, "ev_ebitda": 12},
    "Utilities": {"pe": 14, "ev_ebitda": 7},
    "Default": {"pe": 16, "ev_ebitda": 10}
}

# Database path
DB_PATH = "halal_stocks_database.db"

def retry_on_failure(func, max_retries=3, delay=1):
    """Retry a function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay * (2 ** attempt))
    return None

def fetch_stock_data(ticker, max_retries=3):
    """Fetch stock data with retry logic and rate limiting"""
    def _fetch():
        stock = yf.Ticker(ticker)
        info = stock.info
        time.sleep(0.3)
        return info
    
    try:
        return retry_on_failure(_fetch, max_retries)
    except Exception as e:
        return None

def safe_float(value, default=None):
    """Safely convert value to float"""
    if value is None or value == '' or pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=None):
    """Safely convert value to int"""
    if value is None or value == '' or pd.isna(value):
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def check_business_activity_halal(stock_data):
    """Check if core business activity is halal"""
    reasons = []
    is_halal = True
    
    # Check sector and industry
    sector = stock_data.get('sector', '').lower()
    industry = stock_data.get('industry', '').lower()
    business_summary = stock_data.get('longBusinessSummary', '').lower()
    company_name = stock_data.get('longName', '').lower()
    
    # Check for non-halal keywords
    for keyword in NON_HALAL_KEYWORDS:
        if keyword in sector or keyword in industry or keyword in business_summary or keyword in company_name:
            is_halal = False
            reasons.append(f"❌ Business involves {keyword.title()}")
    
    # Specific sector checks
    if 'financial' in sector or 'bank' in sector or 'insurance' in sector:
        is_halal = False
        reasons.append("❌ Conventional Financial Services (Interest-based)")
    
    if 'tobacco' in sector or 'tobacco' in industry:
        is_halal = False
        reasons.append("❌ Tobacco Industry")
    
    if 'alcohol' in sector or 'beverage' in sector:
        # Need to check if it's alcoholic beverages
        if 'alcohol' in industry or 'brewery' in industry or 'distillery' in industry:
            is_halal = False
            reasons.append("❌ Alcoholic Beverages")
    
    if 'gambling' in sector or 'casino' in industry:
        is_halal = False
        reasons.append("❌ Gambling Industry")
    
    if 'defense' in sector or 'weapons' in industry:
        is_halal = False
        reasons.append("❌ Weapons & Defense Industry")
    
    if is_halal:
        reasons.append("✅ Core business activity is permissible")
    
    return is_halal, reasons

def check_shariah_compliance(stock_data, standard="AAOIFI"):
    """Check if stock meets Shariah compliance criteria"""
    criteria = SHARIAH_STANDARDS[standard]
    reasons = []
    compliance_scores = {}
    is_compliant = True
    
    # Extract financial data
    market_cap = safe_float(stock_data.get('marketCap'))
    total_debt = safe_float(stock_data.get('totalDebt', 0))
    total_cash = safe_float(stock_data.get('totalCash', 0))
    total_revenue = safe_float(stock_data.get('totalRevenue'))
    
    # Calculate ratios
    if market_cap and market_cap > 0:
        # 1. Debt to Market Cap Ratio
        debt_ratio = (total_debt / market_cap) if total_debt else 0
        compliance_scores['debt_ratio'] = debt_ratio
        
        if debt_ratio <= criteria['debt_to_market_cap']:
            reasons.append(f"✅ Debt/Market Cap: {debt_ratio:.2%} (Limit: {criteria['debt_to_market_cap']:.0%})")
        else:
            is_compliant = False
            reasons.append(f"❌ Debt/Market Cap: {debt_ratio:.2%} exceeds {criteria['debt_to_market_cap']:.0%} limit")
        
        # 2. Liquid Assets Ratio (Cash + Interest-bearing securities)
        # Assuming total cash includes interest-bearing securities
        liquid_ratio = (total_cash / market_cap) if total_cash else 0
        compliance_scores['liquid_ratio'] = liquid_ratio
        
        if liquid_ratio <= criteria['liquid_assets_ratio']:
            reasons.append(f"✅ Liquid Assets/Market Cap: {liquid_ratio:.2%} (Limit: {criteria['liquid_assets_ratio']:.0%})")
        else:
            is_compliant = False
            reasons.append(f"❌ Liquid Assets/Market Cap: {liquid_ratio:.2%} exceeds {criteria['liquid_assets_ratio']:.0%} limit")
    else:
        reasons.append("⚠️ Market cap data not available")
        is_compliant = False
    
    # 3. Interest Income Ratio (approximated from financial data)
    # Note: yfinance doesn't directly provide interest income, so we approximate
    if total_revenue and total_revenue > 0:
        # This is an approximation - in real scenario, you'd need detailed financials
        # For now, we'll assume if it's not a financial company, interest income is minimal
        sector = stock_data.get('sector', '').lower()
        if 'financial' in sector:
            reasons.append("⚠️ Financial sector - requires detailed income statement review")
            is_compliant = False
        else:
            reasons.append("✅ Non-financial sector - minimal interest income assumed")
    
    compliance_scores['overall'] = is_compliant
    
    return is_compliant, reasons, compliance_scores

def get_halal_status(stock_data, standard="AAOIFI"):
    """Get overall halal status of a stock"""
    # Check business activity
    business_halal, business_reasons = check_business_activity_halal(stock_data)
    
    # Check Shariah compliance
    shariah_compliant, shariah_reasons, compliance_scores = check_shariah_compliance(stock_data, standard)
    
    # Overall status
    if business_halal and shariah_compliant:
        status = "HALAL"
        status_emoji = "✅"
    elif business_halal and not shariah_compliant:
        status = "DOUBTFUL"
        status_emoji = "⚠️"
    else:
        status = "NON-HALAL"
        status_emoji = "❌"
    
    all_reasons = business_reasons + shariah_reasons
    
    return {
        'status': status,
        'status_emoji': status_emoji,
        'business_halal': business_halal,
        'shariah_compliant': shariah_compliant,
        'reasons': all_reasons,
        'compliance_scores': compliance_scores
    }

def calculate_valuations(stock_data, market_cap_category):
    """Calculate fair value using multiple methods"""
    try:
        industry = stock_data.get('industry', 'Default')
        if industry not in INDUSTRY_BENCHMARKS:
            industry = 'Default'
        
        if market_cap_category == "Large Cap":
            benchmarks = INDUSTRY_BENCHMARKS[industry]
        elif market_cap_category == "Mid Cap":
            benchmarks = MIDCAP_BENCHMARKS[industry]
        else:
            benchmarks = SMALLCAP_BENCHMARKS[industry]
        
        current_price = safe_float(stock_data.get('currentPrice'))
        if not current_price:
            return None, None, None, None
        
        # PE Multiple Method
        trailing_pe = safe_float(stock_data.get('trailingPE'))
        forward_pe = safe_float(stock_data.get('forwardPE'))
        eps = safe_float(stock_data.get('trailingEps'))
        
        pe_fair_value = None
        if eps and eps > 0:
            if trailing_pe:
                historical_fair_pe = trailing_pe * 0.9
                target_pe = (historical_fair_pe * 0.7) + (benchmarks['pe'] * 0.3)
                pe_fair_value = eps * target_pe
            elif forward_pe:
                target_pe = (forward_pe * 0.7) + (benchmarks['pe'] * 0.3)
                pe_fair_value = eps * target_pe
        
        # EV/EBITDA Method
        enterprise_value = safe_float(stock_data.get('enterpriseValue'))
        ebitda = safe_float(stock_data.get('ebitda'))
        market_cap = safe_float(stock_data.get('marketCap'))
        
        ev_ebitda_fair_value = None
        if ebitda and ebitda > 0 and enterprise_value and market_cap:
            current_ev_ebitda = enterprise_value / ebitda
            target_ev_ebitda = (current_ev_ebitda * 0.5) + (benchmarks['ev_ebitda'] * 0.5)
            fair_enterprise_value = ebitda * target_ev_ebitda
            ev_to_mcap_ratio = market_cap / enterprise_value
            ev_ebitda_fair_value = fair_enterprise_value * ev_to_mcap_ratio
        
        fair_values = [v for v in [pe_fair_value, ev_ebitda_fair_value] if v is not None]
        if not fair_values:
            return None, None, None, None
        
        avg_fair_value = sum(fair_values) / len(fair_values)
        upside_potential = ((avg_fair_value - current_price) / current_price) * 100
        
        pe_upside = ((pe_fair_value - current_price) / current_price * 100) if pe_fair_value else None
        ev_upside = ((ev_ebitda_fair_value - current_price) / current_price * 100) if ev_ebitda_fair_value else None
        
        return avg_fair_value, upside_potential, pe_upside, ev_upside
    
    except Exception as e:
        return None, None, None, None

def get_market_cap_category(market_cap):
    """Categorize stock by market cap"""
    if market_cap >= 100000_00_00_000:
        return "Large Cap"
    elif market_cap >= 25000_00_00_000:
        return "Mid Cap"
    elif market_cap >= 5000_00_00_000:
        return "Small Cap"
    else:
        return "Micro Cap"

def create_database():
    """Create SQLite database for storing stock data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS halal_stocks (
            ticker TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL,
            market_cap REAL,
            pe_ratio REAL,
            forward_pe REAL,
            eps REAL,
            pb_ratio REAL,
            dividend_yield REAL,
            beta REAL,
            roe REAL,
            profit_margin REAL,
            revenue REAL,
            ebitda REAL,
            enterprise_value REAL,
            total_debt REAL,
            total_cash REAL,
            shares_outstanding REAL,
            week_52_high REAL,
            week_52_low REAL,
            sector TEXT,
            industry TEXT,
            halal_status TEXT,
            business_halal INTEGER,
            shariah_compliant INTEGER,
            debt_ratio REAL,
            liquid_ratio REAL,
            compliance_reasons TEXT,
            last_updated TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def update_database(stock_dict, standard="AAOIFI", progress_callback=None):
    """Update database with current stock data and halal compliance"""
    create_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    total_stocks = sum(len(stocks) for stocks in stock_dict.values())
    processed = 0
    failed_tickers = []
    halal_count = 0
    doubtful_count = 0
    non_halal_count = 0
    
    for category, stocks in stock_dict.items():
        for ticker, name in stocks.items():
            try:
                stock_data = fetch_stock_data(ticker)
                if stock_data:
                    # Get halal status
                    halal_info = get_halal_status(stock_data, standard)
                    
                    # Count statuses
                    if halal_info['status'] == 'HALAL':
                        halal_count += 1
                    elif halal_info['status'] == 'DOUBTFUL':
                        doubtful_count += 1
                    else:
                        non_halal_count += 1
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO halal_stocks VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    ''', (
                        ticker,
                        name,
                        category,
                        safe_float(stock_data.get('currentPrice')),
                        safe_float(stock_data.get('marketCap')),
                        safe_float(stock_data.get('trailingPE')),
                        safe_float(stock_data.get('forwardPE')),
                        safe_float(stock_data.get('trailingEps')),
                        safe_float(stock_data.get('priceToBook')),
                        safe_float(stock_data.get('dividendYield')),
                        safe_float(stock_data.get('beta')),
                        safe_float(stock_data.get('returnOnEquity')),
                        safe_float(stock_data.get('profitMargins')),
                        safe_float(stock_data.get('totalRevenue')),
                        safe_float(stock_data.get('ebitda')),
                        safe_float(stock_data.get('enterpriseValue')),
                        safe_float(stock_data.get('totalDebt')),
                        safe_float(stock_data.get('totalCash')),
                        safe_float(stock_data.get('sharesOutstanding')),
                        safe_float(stock_data.get('fiftyTwoWeekHigh')),
                        safe_float(stock_data.get('fiftyTwoWeekLow')),
                        stock_data.get('sector', 'N/A'),
                        stock_data.get('industry', 'N/A'),
                        halal_info['status'],
                        1 if halal_info['business_halal'] else 0,
                        1 if halal_info['shariah_compliant'] else 0,
                        halal_info['compliance_scores'].get('debt_ratio', 0),
                        halal_info['compliance_scores'].get('liquid_ratio', 0),
                        ' | '.join(halal_info['reasons']),
                        datetime.now().isoformat()
                    ))
                    conn.commit()
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                failed_tickers.append(ticker)
            
            processed += 1
            if progress_callback:
                progress_callback(processed, total_stocks, failed_tickers, halal_count, doubtful_count, non_halal_count)
    
    conn.close()
    return failed_tickers, halal_count, doubtful_count, non_halal_count

@st.cache_data(ttl=3600)
def load_database():
    """Load stock data from database with proper type conversion"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM halal_stocks", conn)
        conn.close()
        
        # Convert numeric columns explicitly
        numeric_columns = [
            'price', 'market_cap', 'pe_ratio', 'forward_pe', 'eps', 'pb_ratio',
            'dividend_yield', 'beta', 'roe', 'profit_margin', 'revenue', 'ebitda',
            'enterprise_value', 'total_debt', 'total_cash', 'shares_outstanding',
            'week_52_high', 'week_52_low', 'debt_ratio', 'liquid_ratio'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return pd.DataFrame()

def check_database_exists():
    """Check if database exists and is not empty"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM halal_stocks")
        count = cursor.fetchone()[0]
        
        # Get halal counts
        cursor.execute("SELECT halal_status, COUNT(*) FROM halal_stocks GROUP BY halal_status")
        status_counts = dict(cursor.fetchall())
        
        conn.close()
        return count > 0, count, status_counts
    except:
        return False, 0, {}

def parallel_fetch_stocks(stock_list, standard="AAOIFI", max_workers=15):
    """Fetch stock data in parallel with progress bar"""
    results = []
    failed = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(stock_list)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stock = {
            executor.submit(fetch_stock_data, ticker): (ticker, name, category)
            for category, stocks in stock_list.items()
            for ticker, name in stocks.items()
        }
        
        for future in as_completed(future_to_stock):
            ticker, name, category = future_to_stock[future]
            completed += 1
            
            try:
                stock_data = future.result()
                if stock_data:
                    halal_info = get_halal_status(stock_data, standard)
                    results.append({
                        'ticker': ticker,
                        'name': name,
                        'category': category,
                        'data': stock_data,
                        'halal_info': halal_info
                    })
                else:
                    failed.append(ticker)
            except Exception as e:
                failed.append(ticker)
            
            progress_bar.progress(completed / total)
            status_text.text(f"Processed {completed}/{total} stocks ({len(failed)} failed)")
    
    progress_bar.empty()
    status_text.empty()
    
    return results, failed

def screen_from_database(df, criteria):
    """Screen stocks from database based on criteria"""
    filtered_df = df.copy()
    
    # Halal status filter
    if criteria.get('halal_status') and criteria['halal_status'] != "All":
        filtered_df = filtered_df[filtered_df['halal_status'] == criteria['halal_status']]
    
    # Categories filter
    if criteria.get('categories'):
        filtered_df = filtered_df[filtered_df['category'].isin(criteria['categories'])]
    
    # Market cap filter
    if criteria.get('market_cap_category'):
        if criteria['market_cap_category'] == "Large Cap":
            filtered_df = filtered_df[filtered_df['market_cap'] >= 100000_00_00_000]
        elif criteria['market_cap_category'] == "Mid Cap":
            filtered_df = filtered_df[
                (filtered_df['market_cap'] >= 25000_00_00_000) & 
                (filtered_df['market_cap'] < 100000_00_00_000)
            ]
        elif criteria['market_cap_category'] == "Small Cap":
            filtered_df = filtered_df[
                (filtered_df['market_cap'] >= 5000_00_00_000) & 
                (filtered_df['market_cap'] < 25000_00_00_000)
            ]
        elif criteria['market_cap_category'] == "Micro Cap":
            filtered_df = filtered_df[filtered_df['market_cap'] < 5000_00_00_000]
    
    # Price filter
    if criteria.get('min_price'):
        filtered_df = filtered_df[filtered_df['price'] >= criteria['min_price']]
    
    if criteria.get('max_price'):
        filtered_df = filtered_df[filtered_df['price'] <= criteria['max_price']]
    
    # PE filter
    if criteria.get('max_pe'):
        filtered_df = filtered_df[
            (filtered_df['pe_ratio'] <= criteria['max_pe']) & 
            (filtered_df['pe_ratio'] > 0)
        ]
    
    # Debt ratio filter
    if criteria.get('max_debt_ratio'):
        filtered_df = filtered_df[filtered_df['debt_ratio'] <= criteria['max_debt_ratio']]
    
    return filtered_df

def calculate_valuations_batch(df):
    """Calculate valuations for a batch of stocks from database"""
    results = []
    
    for idx, row in df.iterrows():
        stock_data = {
            'currentPrice': row['price'],
            'marketCap': row['market_cap'],
            'trailingPE': row['pe_ratio'],
            'forwardPE': row['forward_pe'],
            'trailingEps': row['eps'],
            'enterpriseValue': row['enterprise_value'],
            'ebitda': row['ebitda'],
            'industry': row['industry']
        }
        
        market_cap_cat = get_market_cap_category(row['market_cap']) if pd.notna(row['market_cap']) else "Micro Cap"
        fair_value, upside, pe_upside, ev_upside = calculate_valuations(stock_data, market_cap_cat)
        
        if fair_value is not None:
            results.append({
                'Ticker': row['ticker'],
                'Name': row['name'],
                'Category': row['category'],
                'Halal Status': row['halal_status'],
                'Current Price': row['price'],
                'Fair Value': fair_value,
                'Upside %': upside,
                'PE Upside %': pe_upside,
                'EV/EBITDA Upside %': ev_upside,
                'Market Cap': row['market_cap'],
                'Market Cap Category': market_cap_cat,
                'PE Ratio': row['pe_ratio'],
                'Debt Ratio': row['debt_ratio'],
                'Liquid Ratio': row['liquid_ratio'],
                '52W High': row['week_52_high'],
                '52W Low': row['week_52_low'],
                'Compliance Reasons': row['compliance_reasons']
            })
    
    return pd.DataFrame(results)

def get_preset_screeners():
    """Define preset halal screeners"""
    return {
        "☪️ Top 50 Halal Large Caps": {
            "halal_status": "HALAL",
            "market_cap_category": "Large Cap",
            "min_upside": 15,
            "limit": 50
        },
        "☪️ Top 50 Halal Mid Caps": {
            "halal_status": "HALAL",
            "market_cap_category": "Mid Cap",
            "min_upside": 20,
            "limit": 50
        },
        "☪️ Top 50 Halal Small Caps": {
            "halal_status": "HALAL",
            "market_cap_category": "Small Cap",
            "min_upside": 25,
            "limit": 50
        },
        "💎 Halal Value Stocks (PE < 15)": {
            "halal_status": "HALAL",
            "max_pe": 15,
            "min_upside": 15,
            "limit": 50
        },
        "🚀 Highly Undervalued Halal Stocks": {
            "halal_status": "HALAL",
            "min_upside": 30,
            "limit": 50
        },
        "⚖️ Low Debt Halal Stocks": {
            "halal_status": "HALAL",
            "max_debt_ratio": 0.15,
            "min_upside": 15,
            "limit": 50
        }
    }

def create_compliance_gauge(score, title, threshold=0.33):
    """Create a gauge chart for compliance metrics"""
    color = "#10b981" if score <= threshold else "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': '#10b981'}},
        number={'suffix': "%", 'font': {'size': 28, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "#10b981"},
            'bar': {'color': color},
            'bgcolor': "rgba(30, 30, 46, 0.5)",
            'borderwidth': 2,
            'bordercolor': "#10b981",
            'steps': [
                {'range': [0, threshold * 100], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [threshold * 100, 50], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(30, 30, 46, 0.2)",
        plot_bgcolor="rgba(30, 30, 46, 0.2)",
        font={'color': "#ffffff", 'family': "Arial"},
        height=250
    )
    
    return fig

def create_valuation_gauge(value, title, range_max=100):
    """Create a gauge chart for valuation"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': '#10b981'}},
        number={'suffix': "%", 'font': {'size': 28, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [None, range_max], 'tickwidth': 1, 'tickcolor': "#10b981"},
            'bar': {'color': "#10b981"},
            'bgcolor': "rgba(30, 30, 46, 0.5)",
            'borderwidth': 2,
            'bordercolor': "#10b981",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [25, 50], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [50, range_max], 'color': 'rgba(16, 185, 129, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(30, 30, 46, 0.2)",
        plot_bgcolor="rgba(30, 30, 46, 0.2)",
        font={'color': "#ffffff", 'family': "Arial"},
        height=250
    )
    
    return fig

def create_bar_chart(current, fair, ticker):
    """Create a bar chart comparing current and fair value"""
    fig = go.Figure(data=[
        go.Bar(name='Current Price', x=['Price'], y=[current], marker_color='#ef4444'),
        go.Bar(name='Fair Value', x=['Price'], y=[fair], marker_color='#10b981')
    ])
    
    fig.update_layout(
        title=f"{ticker} - Price Comparison",
        yaxis_title="Price (₹)",
        barmode='group',
        paper_bgcolor="rgba(30, 30, 46, 0.2)",
        plot_bgcolor="rgba(30, 30, 46, 0.2)",
        font={'color': "#ffffff"},
        title_font_color="#10b981",
        height=350
    )
    
    return fig

def generate_halal_pdf_report(stock_info):
    """Generate detailed Halal compliance PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#059669'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph(f"☪️ Halal Stock Analysis Report", title_style))
    story.append(Paragraph(f"{stock_info['ticker']}", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Company Info
    data = [
        ['Company Name', stock_info['name']],
        ['Ticker', stock_info['ticker']],
        ['Category', stock_info['category']],
        ['Sector', stock_info.get('sector', 'N/A')],
        ['Industry', stock_info.get('industry', 'N/A')],
    ]
    
    t = Table(data, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#059669')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(t)
    story.append(Spacer(1, 0.3*inch))
    
    # Halal Compliance Status
    story.append(Paragraph("Shariah Compliance Status", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    status = stock_info.get('halal_status', 'UNKNOWN')
    status_color = colors.HexColor('#10b981') if status == 'HALAL' else colors.HexColor('#ef4444')
    
    status_data = [
        ['Status', status],
        ['Business Activity', 'Halal' if stock_info.get('business_halal') else 'Non-Halal'],
        ['Financial Compliance', 'Compliant' if stock_info.get('shariah_compliant') else 'Non-Compliant'],
        ['Debt/Market Cap', f"{stock_info.get('debt_ratio', 0):.2%}"],
        ['Liquid Assets/Market Cap', f"{stock_info.get('liquid_ratio', 0):.2%}"],
    ]
    
    t2 = Table(status_data, colWidths=[2.5*inch, 3.5*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), status_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(t2)
    story.append(Spacer(1, 0.3*inch))
    
    # Compliance Reasons
    story.append(Paragraph("Compliance Details", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    reasons = stock_info.get('reasons', [])
    for reason in reasons:
        reason_style = ParagraphStyle(
            'Reason',
            parent=styles['Normal'],
            fontSize=10,
            spaceBefore=5,
            spaceAfter=5
        )
        story.append(Paragraph(reason, reason_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Valuation Analysis
    story.append(Paragraph("Valuation Analysis", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    valuation_data = [
        ['Metric', 'Value'],
        ['Current Price', f"₹{stock_info.get('current_price', 0):.2f}"],
        ['Fair Value', f"₹{stock_info.get('fair_value', 0):.2f}"],
        ['Upside Potential', f"{stock_info.get('upside', 0):.2f}%"],
        ['Market Cap Category', stock_info.get('market_cap_cat', 'N/A')],
        ['PE Ratio', f"{stock_info.get('pe_ratio', 0):.2f}"],
        ['Investment Recommendation', stock_info.get('recommendation', 'N/A')],
    ]
    
    t3 = Table(valuation_data, colWidths=[2.5*inch, 3.5*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#059669')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(t3)
    story.append(Spacer(1, 0.5*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_JUSTIFY
    )
    
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is for educational purposes only and does not constitute "
        "investment advice or Shariah certification. The Shariah compliance assessment is based on "
        "publicly available financial data and general screening criteria. Individual investors should "
        "consult qualified Islamic scholars and financial advisors before making investment decisions. "
        "Compliance status may change over time based on company activities and financial position.",
        disclaimer_style
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Main application

    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 Account")
        st.info(f"User: {st.session_state['user'].title()}")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Shariah Standard Selection
        st.markdown("### ☪️ Shariah Standard")
        selected_standard = st.selectbox(
            "Select Standard",
            list(SHARIAH_STANDARDS.keys()),
            help="Choose the Shariah compliance standard to apply"
        )
        st.info(f"📋 {SHARIAH_STANDARDS[selected_standard]['name']}")
        
        st.markdown("---")
        
        # Database Status
        st.markdown("### 📊 Database Status")
        db_exists, stock_count, status_counts = check_database_exists()
        
        if db_exists:
            st.success(f"✅ Database: {stock_count} stocks")
            
            # Show halal breakdown
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("☪️ Halal", status_counts.get('HALAL', 0))
            with col2:
                st.metric("⚠️ Doubtful", status_counts.get('DOUBTFUL', 0))
            with col3:
                st.metric("❌ Non-Halal", status_counts.get('NON-HALAL', 0))
            
            # Show last updated time
            try:
                df = load_database()
                if not df.empty and 'last_updated' in df.columns:
                    last_updated = pd.to_datetime(df['last_updated']).max()
                    st.info(f"🕒 Updated: {last_updated.strftime('%Y-%m-%d %H:%M')}")
            except:
                pass
        else:
            st.warning("⚠️ No database found")
            st.info("👉 Update database to start screening")
        
        st.markdown("---")
        
        # Database Management
        st.markdown("### ⚙️ Database Management")
        
        total_to_update = sum(len(stocks) for stocks in INDIAN_HALAL_STOCKS.values())
        st.info(f"📌 Will analyze {total_to_update} stocks")
        
        if st.button("🔄 Update Database Now", use_container_width=True):
            st.session_state['show_update_confirmation'] = True
        
        if st.session_state.get('show_update_confirmation', False):
            st.warning("⚠️ This may take 30-60 minutes")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm", use_container_width=True):
                    st.session_state['show_update_confirmation'] = False
                    st.session_state['updating_database'] = True
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state['show_update_confirmation'] = False
                    st.rerun()
    
    # Handle database update
    if st.session_state.get('updating_database', False):
        st.markdown("## 🔄 Updating Database with Halal Compliance Analysis...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        failed_text = st.empty()
        halal_status_text = st.empty()
        
        def progress_callback(processed, total, failed, halal, doubtful, non_halal):
            progress_bar.progress(processed / total)
            status_text.text(f"Processed: {processed}/{total}")
            if failed:
                failed_text.text(f"Failed: {len(failed)} stocks")
            halal_status_text.text(f"☪️ Halal: {halal} | ⚠️ Doubtful: {doubtful} | ❌ Non-Halal: {non_halal}")
        
        failed_tickers, halal_count, doubtful_count, non_halal_count = update_database(
            INDIAN_HALAL_STOCKS, 
            selected_standard, 
            progress_callback
        )
        
        st.session_state['updating_database'] = False
        st.success(f"✅ Database updated!")
        st.info(f"☪️ Halal: {halal_count} | ⚠️ Doubtful: {doubtful_count} | ❌ Non-Halal: {non_halal_count}")
        
        if failed_tickers:
            with st.expander("View Failed Tickers"):
                st.write(failed_tickers)
        
        st.cache_data.clear()
        time.sleep(2)
        st.rerun()
    
    # Main content
    st.markdown("<h1 style='text-align: center;'>☪️ NYZTrade Halal Stock Screener</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #10b981;'>Shariah-Compliant Investment Analysis with Valuation</p>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["☪️ Halal Presets", "🔍 Custom Halal Search", "📈 Individual Analysis", "📚 Compliance Guide"])
    
    # Tab 1: Halal Presets
    with tab1:
        st.markdown("### ⚡ INSTANT HALAL STOCK SCREENING")
        
        if not db_exists:
            st.warning("📊 Please update the database first using the sidebar button")
        else:
            preset_screeners = get_preset_screeners()
            
            st.markdown("### 🎯 Select a Preset Halal Screener")
            selected_preset = st.selectbox(
                "Choose Screener",
                list(preset_screeners.keys()),
                label_visibility="collapsed"
            )
            
            if st.button("🚀 Run Halal Screening", use_container_width=True, type="primary"):
                with st.spinner("Screening Halal stocks..."):
                    df = load_database()
                    
                    if df.empty:
                        st.error("Database is empty. Please update it first.")
                    else:
                        criteria = preset_screeners[selected_preset]
                        
                        # Filter based on criteria
                        filtered_df = screen_from_database(df, {
                            'halal_status': criteria.get('halal_status'),
                            'market_cap_category': criteria.get('market_cap_category'),
                            'max_pe': criteria.get('max_pe'),
                            'max_debt_ratio': criteria.get('max_debt_ratio')
                        })
                        
                        # Calculate valuations
                        results_df = calculate_valuations_batch(filtered_df)
                        
                        # Apply valuation filters
                        if criteria.get('min_upside'):
                            results_df = results_df[results_df['Upside %'] >= criteria['min_upside']]
                        
                        # Sort and limit
                        results_df = results_df.nlargest(criteria.get('limit', 50), 'Upside %')
                        
                        st.success(f"✅ Found {len(results_df)} Halal stocks matching criteria")
                        
                        # Display results
                        if not results_df.empty:
                            display_df = results_df[[
                                'Ticker', 'Name', 'Halal Status', 'Current Price',
                                'Fair Value', 'Upside %', 'Debt Ratio', 'PE Ratio'
                            ]].copy()
                            
                            # Format display
                            display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"₹{x:.2f}")
                            display_df['Fair Value'] = display_df['Fair Value'].apply(lambda x: f"₹{x:.2f}")
                            display_df['Upside %'] = display_df['Upside %'].apply(lambda x: f"{x:.2f}%")
                            display_df['Debt Ratio'] = display_df['Debt Ratio'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                            display_df['PE Ratio'] = display_df['PE Ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                            
                            st.dataframe(display_df, use_container_width=True, height=400)
                            
                            # Download option
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results (CSV)",
                                data=csv,
                                file_name=f"halal_{selected_preset.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
    
    # Tab 2: Custom Halal Search
    with tab2:
        st.markdown("### 🔍 Custom Halal Stock Search")
        st.info("💡 Search with custom filters for Shariah-compliant stocks")
        
        with st.form("custom_halal_search"):
            st.markdown("#### ☪️ Halal Compliance Filters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                halal_status_filter = st.selectbox(
                    "Halal Status",
                    ["All", "HALAL", "DOUBTFUL", "NON-HALAL"]
                )
                
                max_debt_ratio_filter = st.slider(
                    "Max Debt/Market Cap (%)",
                    0, 50, 33, step=1,
                    help="Shariah standard: ≤33%"
                )
            
            with col2:
                max_liquid_ratio_filter = st.slider(
                    "Max Liquid Assets/Market Cap (%)",
                    0, 50, 33, step=1,
                    help="Shariah standard: ≤33%"
                )
            
            st.markdown("#### 📊 Market & Valuation Filters")
            
            col3, col4 = st.columns(2)
            
            with col3:
                categories = st.multiselect(
                    "Select Categories",
                    list(INDIAN_HALAL_STOCKS.keys()),
                    max_selections=10
                )
                
                market_cap_filter = st.selectbox(
                    "Market Cap Category",
                    ["All", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap"]
                )
            
            with col4:
                price_range = st.slider(
                    "Price Range (₹)",
                    0, 10000, (0, 10000), step=100
                )
                
                min_upside_filter = st.slider(
                    "Minimum Upside (%)",
                    -50, 100, 15, step=5
                )
            
            submit_button = st.form_submit_button("🔍 Search Halal Stocks", use_container_width=True, type="primary")
        
        if submit_button:
            if not db_exists:
                st.warning("Please update the database first")
            else:
                with st.spinner("Searching Halal stocks..."):
                    df = load_database()
                    
                    if df.empty:
                        st.error("Database is empty. Please update it first.")
                    else:
                        # Apply filters
                        filtered_df = screen_from_database(df, {
                            'halal_status': halal_status_filter if halal_status_filter != "All" else None,
                            'categories': categories if categories else None,
                            'market_cap_category': market_cap_filter if market_cap_filter != "All" else None,
                            'min_price': price_range[0],
                            'max_price': price_range[1],
                            'max_debt_ratio': max_debt_ratio_filter / 100
                        })
                        
                        # Filter by liquid ratio
                        if max_liquid_ratio_filter < 50:
                            filtered_df = filtered_df[filtered_df['liquid_ratio'] <= max_liquid_ratio_filter / 100]
                        
                        # Calculate valuations
                        results_df = calculate_valuations_batch(filtered_df)
                        
                        # Apply upside filter
                        results_df = results_df[results_df['Upside %'] >= min_upside_filter]
                        
                        # Sort by upside
                        results_df = results_df.sort_values('Upside %', ascending=False)
                        
                        st.success(f"✅ Found {len(results_df)} stocks matching criteria")
                        
                        # Display results
                        if not results_df.empty:
                            display_df = results_df[[
                                'Ticker', 'Name', 'Halal Status', 'Current Price',
                                'Fair Value', 'Upside %', 'Debt Ratio', 'Liquid Ratio', 'PE Ratio'
                            ]].copy()
                            
                            # Format display
                            display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"₹{x:.2f}")
                            display_df['Fair Value'] = display_df['Fair Value'].apply(lambda x: f"₹{x:.2f}")
                            display_df['Upside %'] = display_df['Upside %'].apply(lambda x: f"{x:.2f}%")
                            display_df['Debt Ratio'] = display_df['Debt Ratio'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                            display_df['Liquid Ratio'] = display_df['Liquid Ratio'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                            display_df['PE Ratio'] = display_df['PE Ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                            
                            st.dataframe(display_df, use_container_width=True, height=400)
                            
                            # Download option
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results (CSV)",
                                data=csv,
                                file_name=f"custom_halal_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.warning("No stocks found matching your criteria")
    
    # Tab 3: Individual Analysis
    with tab3:
        st.markdown("### 📈 Individual Halal Stock Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticker_input = st.text_input(
                "Enter Stock Ticker (e.g., RELIANCE.NS)",
                placeholder="TICKER.NS"
            )
        
        with col2:
            analyze_button = st.button("📊 Analyze Stock", use_container_width=True, type="primary")
        
        if analyze_button and ticker_input:
            with st.spinner(f"Analyzing {ticker_input} for Halal compliance..."):
                stock_data = fetch_stock_data(ticker_input)
                
                if stock_data:
                    # Get halal status
                    halal_info = get_halal_status(stock_data, selected_standard)
                    
                    # Display company name and status
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"### {stock_data.get('longName', ticker_input)}")
                    
                    with col2:
                        status = halal_info['status']
                        if status == "HALAL":
                            st.markdown(f"<div class='halal-badge'>{halal_info['status_emoji']} HALAL</div>", unsafe_allow_html=True)
                        elif status == "DOUBTFUL":
                            st.markdown(f"<div class='doubtful-badge'>{halal_info['status_emoji']} DOUBTFUL</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='non-halal-badge'>{halal_info['status_emoji']} NON-HALAL</div>", unsafe_allow_html=True)
                    
                    # Compliance details
                    st.markdown("#### ☪️ Shariah Compliance Analysis")
                    
                    for reason in halal_info['reasons']:
                        if reason.startswith('✅'):
                            st.markdown(f"<div class='compliance-card'>{reason}</div>", unsafe_allow_html=True)
                        elif reason.startswith('❌'):
                            st.markdown(f"<div class='non-compliance-card'>{reason}</div>", unsafe_allow_html=True)
                        else:
                            st.info(reason)
                    
                    # Compliance metrics
                    if halal_info['compliance_scores']:
                        st.markdown("#### 📊 Compliance Metrics")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            debt_ratio = halal_info['compliance_scores'].get('debt_ratio', 0)
                            st.plotly_chart(
                                create_compliance_gauge(debt_ratio, "Debt/Market Cap Ratio", 0.33),
                                use_container_width=True
                            )
                        
                        with col2:
                            liquid_ratio = halal_info['compliance_scores'].get('liquid_ratio', 0)
                            st.plotly_chart(
                                create_compliance_gauge(liquid_ratio, "Liquid Assets Ratio", 0.33),
                                use_container_width=True
                            )
                    
                    # Valuation Analysis
                    current_price = safe_float(stock_data.get('currentPrice'))
                    market_cap = safe_float(stock_data.get('marketCap'))
                    
                    if current_price and market_cap:
                        market_cap_cat = get_market_cap_category(market_cap)
                        fair_value, upside, pe_upside, ev_upside = calculate_valuations(stock_data, market_cap_cat)
                        
                        if fair_value:
                            st.markdown("#### 💰 Valuation Analysis")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Current Price", f"₹{current_price:.2f}")
                            with col2:
                                st.metric("Fair Value", f"₹{fair_value:.2f}")
                            with col3:
                                st.metric("Upside Potential", f"{upside:.2f}%")
                            
                            # Valuation charts
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if pe_upside is not None:
                                    st.plotly_chart(
                                        create_valuation_gauge(pe_upside, "PE Multiple Upside"),
                                        use_container_width=True
                                    )
                            
                            with col2:
                                if ev_upside is not None:
                                    st.plotly_chart(
                                        create_valuation_gauge(ev_upside, "EV/EBITDA Upside"),
                                        use_container_width=True
                                    )
                            
                            # Price comparison
                            st.plotly_chart(
                                create_bar_chart(current_price, fair_value, ticker_input),
                                use_container_width=True
                            )
                            
                            # Investment recommendation
                            if halal_info['status'] == "HALAL":
                                if upside > 25:
                                    recommendation = "🚀 Strong Buy (Halal & Highly Undervalued)"
                                elif upside > 15:
                                    recommendation = "✅ Buy (Halal & Undervalued)"
                                elif upside > 0:
                                    recommendation = "📥 Hold (Halal & Fairly Valued)"
                                else:
                                    recommendation = "⏸️ Hold (Halal but Overvalued)"
                            elif halal_info['status'] == "DOUBTFUL":
                                recommendation = "⚠️ Caution (Requires further Shariah review)"
                            else:
                                recommendation = "❌ Avoid (Not Shariah Compliant)"
                            
                            st.info(f"**Investment Recommendation:** {recommendation}")
                            
                            # Additional info
                            with st.expander("📋 Detailed Information"):
                                info_df = pd.DataFrame({
                                    'Metric': [
                                        'Sector', 'Industry', 'Market Cap', 'Market Cap Category',
                                        'PE Ratio', 'Forward PE', 'PB Ratio', 'Dividend Yield',
                                        'Beta', 'ROE', '52W High', '52W Low'
                                    ],
                                    'Value': [
                                        stock_data.get('sector', 'N/A'),
                                        stock_data.get('industry', 'N/A'),
                                        f"₹{market_cap/10000000:.2f} Cr",
                                        market_cap_cat,
                                        f"{stock_data.get('trailingPE', 0):.2f}",
                                        f"{stock_data.get('forwardPE', 0):.2f}",
                                        f"{stock_data.get('priceToBook', 0):.2f}",
                                        f"{(stock_data.get('dividendYield', 0) * 100):.2f}%",
                                        f"{stock_data.get('beta', 0):.2f}",
                                        f"{(stock_data.get('returnOnEquity', 0) * 100):.2f}%",
                                        f"₹{stock_data.get('fiftyTwoWeekHigh', 0):.2f}",
                                        f"₹{stock_data.get('fiftyTwoWeekLow', 0):.2f}"
                                    ]
                                })
                                st.dataframe(info_df, use_container_width=True, hide_index=True)
                            
                            # PDF Report
                            if st.button("📄 Generate Halal Compliance Report (PDF)", use_container_width=True):
                                pdf_data = {
                                    'ticker': ticker_input,
                                    'name': stock_data.get('longName', ticker_input),
                                    'category': 'N/A',
                                    'sector': stock_data.get('sector', 'N/A'),
                                    'industry': stock_data.get('industry', 'N/A'),
                                    'halal_status': halal_info['status'],
                                    'business_halal': halal_info['business_halal'],
                                    'shariah_compliant': halal_info['shariah_compliant'],
                                    'reasons': halal_info['reasons'],
                                    'debt_ratio': halal_info['compliance_scores'].get('debt_ratio', 0),
                                    'liquid_ratio': halal_info['compliance_scores'].get('liquid_ratio', 0),
                                    'current_price': current_price,
                                    'fair_value': fair_value,
                                    'upside': upside,
                                    'market_cap_cat': market_cap_cat,
                                    'pe_ratio': stock_data.get('trailingPE', 0),
                                    'recommendation': recommendation
                                }
                                
                                pdf_buffer = generate_halal_pdf_report(pdf_data)
                                st.download_button(
                                    label="📥 Download Halal Compliance Report",
                                    data=pdf_buffer,
                                    file_name=f"{ticker_input}_halal_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                        else:
                            st.error("Could not calculate valuation for this stock")
                    else:
                        st.error("Could not fetch price or market cap data")
                else:
                    st.error("Stock not found or data unavailable")
    
    # Tab 4: Compliance Guide
    with tab4:
        st.markdown("### 📚 Shariah Compliance Guide")
        
        st.markdown("""
        #### ☪️ Understanding Halal Stock Investing
        
        Halal (permissible) investing follows Islamic principles that prohibit certain types of 
        businesses and financial practices. This screener helps identify stocks that comply with 
        Shariah guidelines.
        
        ---
        
        #### 🚫 Non-Halal Business Activities
        
        Companies involved in the following activities are considered **Non-Halal**:
        
        1. **Alcohol Production & Distribution** - Intoxicants are prohibited
        2. **Tobacco** - Harmful substances
        3. **Gambling & Casinos** - Games of chance
        4. **Pork Production** - Prohibited meat
        5. **Conventional Banking & Finance** - Interest-based transactions (Riba)
        6. **Conventional Insurance** - Uncertainty (Gharar) in contracts
        7. **Weapons & Defense** - Prohibited unless defensive
        8. **Adult Entertainment** - Immoral content
        
        ---
        
        #### 📊 Financial Compliance Criteria (AAOIFI Standards)
        
        Even if a company's core business is halal, it must meet these financial ratios:
        
        **1. Debt to Market Cap Ratio ≤ 33%**
        - Measures total interest-bearing debt
        - Formula: (Total Debt / Market Capitalization) × 100
        - ✅ Pass: ≤ 33%
        - ❌ Fail: > 33%
        
        **2. Liquid Assets to Market Cap Ratio ≤ 33%**
        - Measures cash and interest-bearing securities
        - Formula: (Cash + Interest-bearing Securities / Market Cap) × 100
        - ✅ Pass: ≤ 33%
        - ❌ Fail: > 33%
        
        **3. Interest Income to Total Revenue ≤ 5%**
        - Non-compliant income from interest
        - Formula: (Interest Income / Total Revenue) × 100
        - ✅ Pass: ≤ 5%
        - ❌ Fail: > 5%
        
        **4. Non-Compliant Income to Total Revenue ≤ 5%**
        - Income from non-halal activities
        - Formula: (Non-Compliant Income / Total Revenue) × 100
        - ✅ Pass: ≤ 5%
        - ❌ Fail: > 5%
        
        ---
        
        #### 🎯 Compliance Status Explained
        
        **✅ HALAL**
        - Core business is permissible
        - Meets all financial ratios
        - Suitable for Shariah-compliant investment
        
        **⚠️ DOUBTFUL**
        - Core business is permissible
        - Fails one or more financial ratios
        - Requires further review or purification
        
        **❌ NON-HALAL**
        - Core business is not permissible
        - Not suitable for Islamic investment
        
        ---
        
        #### 💰 Income Purification
        
        If a stock is **DOUBTFUL** or has minimal non-compliant income (< 5%), you may invest 
        but must **purify** your returns by donating the proportional non-compliant income to charity.
        
        **Formula:**
        ```
        Purification Amount = (Non-Compliant Income / Total Revenue) × Your Total Returns
        ```
        
        ---
        
        #### 📋 Different Shariah Standards
        
        This screener supports multiple standards:
        
        1. **AAOIFI** - Most widely recognized
        2. **DJIM** - Dow Jones Islamic Market Index
        3. **S&P** - S&P Shariah Indices
        
        All use similar criteria with minor variations.
        
        ---
        
        #### ⚠️ Important Disclaimers
        
        1. **Not a Fatwa**: This tool provides screening based on common criteria but is not 
           a religious ruling (fatwa)
        
        2. **Consult Scholars**: Always consult qualified Islamic scholars for specific guidance
        
        3. **Dynamic Status**: Compliance status can change as company activities and 
           finances change
        
        4. **Due Diligence**: Perform your own research beyond automated screening
        
        5. **Data Limitations**: Based on publicly available data which may have delays or 
           inaccuracies
        
        ---
        
        #### 📖 Additional Resources
        
        - AAOIFI Standards: https://aaoifi.com
        - Islamic Finance Council: https://islamicfinancecouncil.org
        - Shariah Board: Consult local Shariah advisory boards
        
        ---
        
        #### 🤲 May Allah Accept Your Halal Earnings
        
        *"O you who believe! Eat of the good things that We have provided for you and be grateful to Allah"*  
        *— Quran 2:172*
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #6b7280;'>"
        "☪️ This tool is for educational purposes only and does not constitute religious or investment advice. "
        "Consult qualified Islamic scholars and financial advisors before making investment decisions. "
        "Shariah compliance status may change over time."
        "</p>",
        unsafe_allow_html=True
    )

# INDIAN_HALAL_STOCKS dictionary - USER MUST POPULATE THIS
# Add your stock universe here

INDIAN_HALAL_STOCKS = {
    # ==================== IT SERVICES ====================
    "IT Services": {
        "TCS.NS": "Tata Consultancy Services Limited",
        "INFY.NS": "Infosys Limited",
        "WIPRO.NS": "Wipro Limited",
        "HCLTECH.NS": "HCL Technologies Limited",
        "TECHM.NS": "Tech Mahindra Limited",
        "LTIM.NS": "LTIMindtree Limited",
        "COFORGE.NS": "Coforge Limited",
        "PERSISTENT.NS": "Persistent Systems Limited",
        "MPHASIS.NS": "Mphasis Limited",
        # Add more IT services stocks below (don't forget commas!)
    },
    
    # ==================== PHARMACEUTICALS ====================
    "Pharmaceuticals": {
        "SUNPHARMA.NS": "Sun Pharmaceutical Industries Limited",
        "DRREDDY.NS": "Dr Reddys Laboratories Limited",
        "CIPLA.NS": "Cipla Limited",
        "BIOCON.NS": "Biocon Limited",
        "DIVISLAB.NS": "Divis Laboratories Limited",
        "AUROPHARMA.NS": "Aurobindo Pharma Limited",
        "TORNTPHARM.NS": "Torrent Pharmaceuticals Limited",
        "ALKEM.NS": "Alkem Laboratories Limited",
        "LUPIN.NS": "Lupin Limited",
        # Add more pharma stocks below
    },
    
    # ==================== FMCG ====================
    "FMCG": {
        "HINDUNILVR.NS": "Hindustan Unilever Limited",
        "ITC.NS": "ITC Limited",
        "NESTLEIND.NS": "Nestle India Limited",
        "BRITANNIA.NS": "Britannia Industries Limited",
        "DABUR.NS": "Dabur India Limited",
        "MARICO.NS": "Marico Limited",
        "GODREJCP.NS": "Godrej Consumer Products Limited",
        "COLPAL.NS": "Colgate Palmolive India Limited",
        "EMAMILTD.NS": "Emami Limited",
        # Add more FMCG stocks below
    },
    
    # ==================== HEALTHCARE ====================
    "Healthcare": {
        "APOLLOHOSP.NS": "Apollo Hospitals Enterprise Limited",
        "FORTIS.NS": "Fortis Healthcare Limited",
        "MAXHEALTH.NS": "Max Healthcare Institute Limited",
        "APOLLOTYRE.NS": "Apollo Tyres Limited",
        "ISGEC.NS": "Isgec Heavy Engineering Limited",
        # Add more healthcare stocks below
    },
    
    # ==================== AUTOMOBILES ====================
    "Automobiles": {
        "MARUTI.NS": "Maruti Suzuki India Limited",
        "M&M.NS": "Mahindra and Mahindra Limited",
        "TATAMOTORS.NS": "Tata Motors Limited",
        "BAJAJ-AUTO.NS": "Bajaj Auto Limited",
        "HEROMOTOCO.NS": "Hero MotoCorp Limited",
        "EICHERMOT.NS": "Eicher Motors Limited",
        "TVSMOTOR.NS": "TVS Motor Company Limited",
        "ASHOKLEY.NS": "Ashok Leyland Limited",
        # Add more auto stocks below
    },
    
    # ==================== CEMENT ====================
    "Cement": {
        "ULTRACEMCO.NS": "UltraTech Cement Limited",
        "AMBUJACEM.NS": "Ambuja Cements Limited",
        "ACC.NS": "ACC Limited",
        "SHREECEM.NS": "Shree Cement Limited",
        "DALMIACEM.NS": "Dalmia Bharat Limited",
        "RAMCOCEM.NS": "Ramco Cements Limited",
        "JKCEMENT.NS": "JK Cement Limited",
        # Add more cement stocks below
    },
    
    # ==================== METALS & MINING ====================
    "Metals & Mining": {
        "HINDALCO.NS": "Hindalco Industries Limited",
        "VEDL.NS": "Vedanta Limited",
        "JSWSTEEL.NS": "JSW Steel Limited",
        "TATASTEEL.NS": "Tata Steel Limited",
        "SAIL.NS": "Steel Authority of India Limited",
        "JINDALSTEL.NS": "Jindal Steel and Power Limited",
        "NMDC.NS": "NMDC Limited",
        "COALINDIA.NS": "Coal India Limited",
        # Add more metals stocks below
    },
    
    # ==================== OIL & GAS ====================
    "Oil & Gas": {
        "RELIANCE.NS": "Reliance Industries Limited",
        "ONGC.NS": "Oil and Natural Gas Corporation Limited",
        "IOC.NS": "Indian Oil Corporation Limited",
        "BPCL.NS": "Bharat Petroleum Corporation Limited",
        "GAIL.NS": "GAIL India Limited",
        "HINDPETRO.NS": "Hindustan Petroleum Corporation Limited",
        "PETRONET.NS": "Petronet LNG Limited",
        # Add more oil & gas stocks below
    },
    
    # ==================== POWER & UTILITIES ====================
    "Power & Utilities": {
        "NTPC.NS": "NTPC Limited",
        "POWERGRID.NS": "Power Grid Corporation of India Limited",
        "ADANIGREEN.NS": "Adani Green Energy Limited",
        "TATAPOWER.NS": "Tata Power Company Limited",
        "TORNTPOWER.NS": "Torrent Power Limited",
        "NHPC.NS": "NHPC Limited",
        # Add more power stocks below
    },
    
    # ==================== TELECOM ====================
    "Telecom": {
        "BHARTIARTL.NS": "Bharti Airtel Limited",
        # Note: Many telecom companies may not be Shariah compliant
        # due to high debt ratios - verify before adding
    },
    
    # ==================== REAL ESTATE ====================
    "Real Estate": {
        "DLF.NS": "DLF Limited",
        "GODREJPROP.NS": "Godrej Properties Limited",
        "OBEROIRLTY.NS": "Oberoi Realty Limited",
        "PRESTIGE.NS": "Prestige Estates Projects Limited",
        "BRIGADE.NS": "Brigade Enterprises Limited",
        # Add more real estate stocks below
    },
    
    # ==================== TEXTILES ====================
    "Textiles": {
        "ARVIND.NS": "Arvind Limited",
        "RAYMOND.NS": "Raymond Limited",
        "AIAENG.NS": "AIA Engineering Limited",
        "WELSPUNIND.NS": "Welspun India Limited",
        # Add more textile stocks below
    },
    
    # ==================== INFRASTRUCTURE ====================
    "Infrastructure": {
        "L&T.NS": "Larsen and Toubro Limited",
        "ADANIPORTS.NS": "Adani Ports and Special Economic Zone Limited",
        "GRASIM.NS": "Grasim Industries Limited",
        "ABB.NS": "ABB India Limited",
        "SIEMENS.NS": "Siemens Limited",
        "BHEL.NS": "Bharat Heavy Electricals Limited",
        # Add more infrastructure stocks below
    },
    
    # ==================== CHEMICALS ====================
    "Chemicals": {
        "UPL.NS": "UPL Limited",
        "PIDILITIND.NS": "Pidilite Industries Limited",
        "ATUL.NS": "Atul Limited",
        "BALRAMCHIN.NS": "Balrampur Chini Mills Limited",
        "DEEPAKNTR.NS": "Deepak Nitrite Limited",
        # Add more chemical stocks below
    },
    
    # ==================== RETAIL ====================
    "Retail": {
        "DMART.NS": "Avenue Supermarts Limited",
        "TRENT.NS": "Trent Limited",
        "SHOPERSTOP.NS": "Shoppers Stop Limited",
        "VMART.NS": "V-Mart Retail Limited",
        # Add more retail stocks below
    },
    
    # ==================== MEDIA & ENTERTAINMENT ====================
    "Media & Entertainment": {
        "PVRINOX.NS": "PVR INOX Limited",
        "NAZARA.NS": "Nazara Technologies Limited",
        "ZEEL.NS": "Zee Entertainment Enterprises Limited",
        # Add more media stocks below
        # Note: Verify content for Shariah compliance
    },
    
    # ==================== LOGISTICS ====================
    "Logistics": {
        "BLUEDART.NS": "Blue Dart Express Limited",
        "VRL.NS": "VRL Logistics Limited",
        "GATI.NS": "Gati Limited",
        "MAHLOG.NS": "Mahindra Logistics Limited",
        # Add more logistics stocks below
    },
    
    # ==================== AGRICULTURE ====================
    "Agriculture": {
        "COROMANDEL.NS": "Coromandel International Limited",
        "KSCL.NS": "Kaveri Seed Company Limited",
        "RALLIS.NS": "Rallis India Limited",
        "GNFC.NS": "Gujarat Narmada Valley Fertilizers and Chemicals Limited",
        # Add more agriculture stocks below
    },
    
    # ==================== ELECTRICAL EQUIPMENT ====================
    "Electrical Equipment": {
        "HAVELLS.NS": "Havells India Limited",
        "CROMPTON.NS": "Crompton Greaves Consumer Electricals Limited",
        "VOLTAS.NS": "Voltas Limited",
        "BLUESTARCO.NS": "Blue Star Limited",
        # Add more electrical equipment stocks below
    },
    
    # ==================== PACKAGING ====================
    "Packaging": {
        "UFLEX.NS": "Uflex Limited",
        "TCNSBRANDS.NS": "TCNS Clothing Company Limited",
        "MAXVENTURE.NS": "Max Ventures and Industries Limited",
        # Add more packaging stocks below
    },
    
    # ==================== EDUCATION ====================
    "Education": {
        "APTECH.NS": "Aptech Limited",
        "MTTECH.NS": "MT Educare Limited",
        "NIITLTD.NS": "NIIT Limited",
        # Add more education stocks below
    },
    
    # ==================== MISCELLANEOUS ====================
    "Miscellaneous": {
        # Add stocks that don't fit other categories
    },

def get_categories():
    """Get list of all available stock categories"""
    return list(INDIAN_STOCKS.keys())

if __name__ == "__main__":
    .main()


