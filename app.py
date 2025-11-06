import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from database import InvestmentDatabase
from data_preprocessing import (load_and_preprocess_data, compute_spending_features,prepare_ml_features,create_sample_users_dataset)
from user_profiling import UserProfiler, build_user_profiles
from allocation_engine import AllocationEngine, check_batch_trigger
from portfolio_simulator import MarketDataSimulator, PortfolioSimulator

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

running_in_cloud = os.environ.get("STREAMLIT_RUNTIME") is not None
st.set_page_config(
    page_title="Smart Investment Round-Up",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.db = InvestmentDatabase()
    st.session_state.market = MarketDataSimulator(use_real_data=True)  # Enable real-time data
    st.session_state.portfolio_sim = PortfolioSimulator(st.session_state.market)
    st.session_state.allocator = AllocationEngine()
    st.session_state.profiler_kmeans = UserProfiler(model_type='kmeans')
    st.session_state.profiler_gmm = UserProfiler(model_type='gmm')
    st.session_state.transactions_df = None
    st.session_state.selected_user = None
    st.session_state.active_model = 'kmeans'
    st.session_state.last_price_update = datetime.now()

def load_data():
    if running_in_cloud:
        # Load from Google Drive link
        url = "https://drive.google.com/uc?id=1WFVwcUXcvWc3BFBkBI1ZYSxk1iAZudUL"
        st.info("Loading data from Google Drive...")
        df = pd.read_csv(url)
    else:
        # Load from local file
        st.info("Loading data from local disk...")
        df = pd.read_csv("data/raw/transactions.csv")
    return df

df = load_data()

st.success(f"Loaded {len(df):,} records successfully!")
st.dataframe(df.head())

def initialize_system():
    """Initialize the system with sample data"""
    with st.spinner("Loading and processing data..."):
        # Load transaction data
        if os.path.exists('data/raw/transactions.csv'):
            df = load_and_preprocess_data('data/raw/transactions.csv', num_users=6)
            st.session_state.transactions_df = df
            
            # Create users in database
            user_ids = create_sample_users_dataset(df, st.session_state.db)
            
            # Load historical transactions into database (sample - last 20 transactions per user)
            for user_id in user_ids:
                user_trans = df[df['user_id'] == user_id].tail(20)
                for _, trans in user_trans.iterrows():
                    st.session_state.db.add_transaction(
                        user_id,
                        trans['amount'],
                        trans['merchant'],
                        trans['category'],
                        trans['timestamp']
                    )
            
            # Build user profiles with K-Means
            st.session_state.profiler_kmeans = build_user_profiles(
                df, user_ids, st.session_state.db, model_type='kmeans'
            )
            
            # Build user profiles with GMM
            st.session_state.profiler_gmm = build_user_profiles(
                df, user_ids, st.session_state.db, model_type='gmm'
            )
            
            st.session_state.initialized = True
            st.success(f"✅ System initialized with {len(user_ids)} users!")
            st.info("📊 Both K-Means and Gaussian Mixture Models trained for comparison!")
        else:
            st.error("❌ Please place your transactions.csv file in data/raw/ folder")

def add_new_transaction(user_id, amount, merchant, category):
    """Add a new transaction and check for batch investment"""
    trans = st.session_state.db.add_transaction(
        user_id, amount, merchant, category
    )
    
    st.success(f"✅ Transaction added! Spare change: ₹{trans['spare_change']:.2f}")
    
    # Check if batch trigger met
    if check_batch_trigger(st.session_state.db, user_id):
        st.info("🎯 Wallet threshold reached! Triggering investment...")
        execute_batch_investment(user_id)

def execute_batch_investment(user_id):
    """Execute a batch investment for user"""
    db = st.session_state.db
    wallet_balance = db.get_wallet_balance(user_id)
    
    if wallet_balance < db.users[user_id]['threshold']:
        st.warning("Wallet balance below threshold. No investment made.")
        return
    
    # Get user profile
    profile = db.users[user_id]['profile']
    
    # Get allocation
    allocation = st.session_state.allocator.get_allocation(
        user_id, profile, wallet_balance
    )
    
    # Execute investment
    result = st.session_state.portfolio_sim.execute_investment(
        db, user_id, allocation
    )
    
    if result:
        st.success(f"✅ Investment executed! Amount: ₹{result['amount']:.2f}")
        st.info(f"Fees: ₹{result['fees']:.2f} | Net Investment: ₹{result['net_investment']:.2f}")
        
        with st.expander("View Allocation Details"):
            for asset, amount in result['allocation'].items():
                st.write(f"• {asset.capitalize()}: ₹{amount:.2f}")
    else:
        st.error("Investment failed - insufficient wallet balance")

def update_ml_allocations():
    """Update ML-based allocations for all users"""
    db = st.session_state.db
    changes_detected = []
    
    with st.spinner("🔄 Updating ML models and checking for changes..."):
        # Step 1: Update allocations based on performance
        for user_id in db.users.keys():
            performance = st.session_state.portfolio_sim.get_asset_performance(
                db, user_id, days=30
            )
            
            if performance:
                profile = db.users[user_id]['profile']
                old_weights = st.session_state.allocator.baseline_allocations[profile].copy()
                
                new_weights, adjustments = st.session_state.allocator.update_weights(
                    user_id, profile, performance
                )
                
                # Check if significant changes occurred
                for asset, adjustment in adjustments.items():
                    if abs(adjustment) > 0.01:  # More than 1% change
                        changes_detected.append({
                            'user': db.users[user_id]['name'],
                            'asset': asset,
                            'old': old_weights.get(asset, 0) * 100,
                            'new': new_weights.get(asset, 0) * 100,
                            'change': adjustment * 100
                        })
    
    # Step 2: Re-profile users based on NEW transactions
    if st.session_state.transactions_df is not None:
        with st.spinner("📊 Re-analyzing user spending patterns..."):
            df = st.session_state.transactions_df
            
            # Add recent transactions from database
            for trans in db.transactions[-100:]:  # Last 100 transactions
                if trans['user_id'] not in df['user_id'].values or \
                   trans['timestamp'] > df['timestamp'].max():
                    new_row = pd.DataFrame([{
                        'user_id': trans['user_id'],
                        'timestamp': trans['timestamp'],
                        'amount': trans['amount'],
                        'merchant': trans['merchant'],
                        'category': trans['category'],
                        'name': db.users[trans['user_id']]['name']
                    }])
                    df = pd.concat([df, new_row], ignore_index=True)
            
            st.session_state.transactions_df = df
            
            # Rebuild profiles
            user_ids = list(db.users.keys())
            
            # Store old profiles
            old_profiles = {}
            for uid in user_ids:
                old_profiles[uid] = {
                    'kmeans': st.session_state.profiler_kmeans.kmeans_profiles.get(uid, {}),
                    'gmm': st.session_state.profiler_gmm.gmm_profiles.get(uid, {})
                }
            
            # Retrain both models
            from src.user_profiling import build_user_profiles
            st.session_state.profiler_kmeans = build_user_profiles(
                df, user_ids, db, model_type='kmeans'
            )
            st.session_state.profiler_gmm = build_user_profiles(
                df, user_ids, db, model_type='gmm'
            )
            
            # Detect cluster changes
            cluster_changes = []
            for uid in user_ids:
                old_km = old_profiles[uid]['kmeans']
                new_km = st.session_state.profiler_kmeans.kmeans_profiles.get(uid, {})
                
                if old_km and new_km:
                    if old_km.get('cluster') != new_km.get('cluster'):
                        cluster_changes.append({
                            'user': db.users[uid]['name'],
                            'model': 'K-Means',
                            'old_cluster': old_km.get('cluster'),
                            'new_cluster': new_km.get('cluster'),
                            'old_profile': old_km.get('profile'),
                            'new_profile': new_km.get('profile'),
                            'old_risk': old_km.get('risk_score', 0),
                            'new_risk': new_km.get('risk_score', 0)
                        })
                
                old_gm = old_profiles[uid]['gmm']
                new_gm = st.session_state.profiler_gmm.gmm_profiles.get(uid, {})
                
                if old_gm and new_gm:
                    if old_gm.get('cluster') != new_gm.get('cluster'):
                        cluster_changes.append({
                            'user': db.users[uid]['name'],
                            'model': 'GMM',
                            'old_cluster': old_gm.get('cluster'),
                            'new_cluster': new_gm.get('cluster'),
                            'old_profile': old_gm.get('profile'),
                            'new_profile': new_gm.get('profile'),
                            'old_risk': old_gm.get('risk_score', 0),
                            'new_risk': new_gm.get('risk_score', 0)
                        })
    
    # Display results
    st.success("✅ ML models updated successfully!")
    
    if cluster_changes:
        st.warning(f"🔄 {len(cluster_changes)} cluster change(s) detected!")
        
        for change in cluster_changes:
            with st.expander(f"🎯 {change['user']} - {change['model']} Profile Changed"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Before:**")
                    st.info(f"Cluster {change['old_cluster']}")
                    st.info(f"{change['old_profile']}")
                    st.metric("Risk Score", f"{change['old_risk']:.2f}")
                
                with col2:
                    st.markdown("**→**")
                    st.markdown("### ➡️")
                
                with col3:
                    st.markdown("**After:**")
                    st.success(f"Cluster {change['new_cluster']}")
                    st.success(f"{change['new_profile']}")
                    st.metric("Risk Score", f"{change['new_risk']:.2f}",
                             delta=f"{change['new_risk'] - change['old_risk']:+.2f}")
                
                # Show what this means for allocation
                old_alloc = st.session_state.allocator.baseline_allocations.get(
                    change['old_profile'], {}
                )
                new_alloc = st.session_state.allocator.baseline_allocations.get(
                    change['new_profile'], {}
                )
                
                st.markdown("**Allocation Impact:**")
                alloc_comparison = pd.DataFrame({
                    'Asset': ['Equity', 'Gold', 'FD', 'Liquid'],
                    'Old %': [
                        old_alloc.get('equity', 0) * 100,
                        old_alloc.get('gold', 0) * 100,
                        old_alloc.get('fd', 0) * 100,
                        old_alloc.get('liquid', 0) * 100
                    ],
                    'New %': [
                        new_alloc.get('equity', 0) * 100,
                        new_alloc.get('gold', 0) * 100,
                        new_alloc.get('fd', 0) * 100,
                        new_alloc.get('liquid', 0) * 100
                    ]
                })
                alloc_comparison['Change'] = alloc_comparison['New %'] - alloc_comparison['Old %']
                st.dataframe(alloc_comparison, use_container_width=True)
    else:
        st.info("ℹ️ No cluster changes detected. Users remain in their current profiles.")
    
    if changes_detected:
        st.info(f"💡 {len(changes_detected)} allocation adjustment(s) made based on performance")
        
        changes_df = pd.DataFrame(changes_detected)
        changes_df['old'] = changes_df['old'].apply(lambda x: f"{x:.1f}%")
        changes_df['new'] = changes_df['new'].apply(lambda x: f"{x:.1f}%")
        changes_df['change'] = changes_df['change'].apply(lambda x: f"{x:+.2f}%")
        changes_df.columns = ['User', 'Asset', 'Old %', 'New %', 'Change']
        
        st.dataframe(changes_df, use_container_width=True)

# Main UI
st.title("💰 Smart Investment Round-Up System")
st.markdown("*Turn your spare change into smart investments using ML*")

# Sidebar
with st.sidebar:
    st.header("🔧 System Control")
    
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", type="primary"):
            initialize_system()
    else:
        st.success("✅ System Active")
        
        # Model selector
        st.divider()
        st.header("🤖 ML Model Selection")
        st.session_state.active_model = st.radio(
            "Active Model:",
            ['kmeans', 'gmm'],
            format_func=lambda x: 'K-Means Clustering' if x == 'kmeans' else 'Gaussian Mixture Model',
            index=0 if st.session_state.active_model == 'kmeans' else 1
        )
        
        # Update ML button
        if st.button("🔄 Update ML Models", type="primary"):
            update_ml_allocations()
                
        st.divider()
        
        # User selection
        st.header("👤 Select User")
        user_list = list(st.session_state.db.users.keys())
        
        # Create better display names
        user_display_names = []
        for uid in user_list:
            name = st.session_state.db.users[uid]['name']
            profile = st.session_state.db.users[uid].get('profile', 'Unknown')
            
            # Get cluster info
            if st.session_state.active_model == 'kmeans':
                cluster_data = st.session_state.profiler_kmeans.kmeans_profiles.get(uid, {})
            else:
                cluster_data = st.session_state.profiler_gmm.gmm_profiles.get(uid, {})
            
            cluster = cluster_data.get('cluster', 0)
            display_name = f"{name} | {profile} | Cluster {cluster}"
            user_display_names.append(display_name)
        
        selected_idx = st.selectbox(
            "Choose User",
            range(len(user_list)),
            format_func=lambda i: user_display_names[i]
        )
        st.session_state.selected_user = user_list[selected_idx]

# Main content
if not st.session_state.initialized:
    st.info("👈 Click 'Initialize System' in the sidebar to load data and start")
    st.markdown("""
    ### Setup Instructions:
    1. Place your `transactions.csv` file in `data/raw/` folder
    2. Click 'Initialize System' button
    3. Select a user from the sidebar
    4. Start adding transactions and see the magic! ✨
    
    ### Features:
    - 🤖 Two ML models (K-Means vs GMM) for comparison
    - 🔍 Explainability (XAI) - understand why you got your profile
    - 📊 Interactive portfolio tracking
    - 💡 Smart allocation with continuous learning
    """)
else:
    user_id = st.session_state.selected_user
    db = st.session_state.db
    
    if user_id:
        user_info = db.users[user_id]
        
        # Get profiles from both models
        kmeans_profile = st.session_state.profiler_kmeans.kmeans_profiles.get(user_id, {})
        gmm_profile = st.session_state.profiler_gmm.gmm_profiles.get(user_id, {})
        
        # Use active model
        if st.session_state.active_model == 'kmeans':
            active_profile = kmeans_profile
        else:
            active_profile = gmm_profile
        
        # User header
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("User", user_info['name'])
        with col2:
            st.metric("Active Model", "K-Means" if st.session_state.active_model == 'kmeans' else "GMM")
        with col3:
            st.metric("Risk Profile", active_profile.get('profile', 'Moderate'))
        with col4:
            st.metric("Risk Score", f"{active_profile.get('risk_score', 0.5):.2f}")
        with col5:
            cluster = active_profile.get('cluster', 0)
            st.metric("Cluster", f"C{cluster}")
        
        # Live market prices display
        st.divider()
        st.markdown("### 📊 Live Market Prices")
        
        # Refresh prices button
        col_refresh1, col_refresh2 = st.columns([3, 1])
        with col_refresh2:
            if st.button("🔄 Refresh Prices"):
                # Store old prices
                old_prices = st.session_state.market.current_prices.copy()
                
                # Fetch new prices
                st.session_state.market._fetch_real_prices()
                st.session_state.last_price_update = datetime.now()
                
                # Check for significant price changes
                new_prices = st.session_state.market.current_prices
                price_changes = []
                
                for asset, new_price in new_prices.items():
                    old_price = old_prices[asset]
                    change_pct = ((new_price - old_price) / old_price) * 100
                    
                    if abs(change_pct) > 1.0:  # More than 1% change
                        price_changes.append({
                            'asset': asset,
                            'old': old_price,
                            'new': new_price,
                            'change': change_pct
                        })
                
                if price_changes:
                    st.warning(f"⚠️ {len(price_changes)} significant price change(s) detected!")
                    
                    for change in price_changes:
                        direction = "📈" if change['change'] > 0 else "📉"
                        st.info(f"{direction} **{change['asset'].capitalize()}**: "
                               f"₹{change['old']:.2f} → ₹{change['new']:.2f} "
                               f"({change['change']:+.2f}%)")
                    
                    # Check if rebalancing is needed
                    portfolio = db.get_portfolio(user_id)
                    needs_rebalance = False
                    
                    for asset, holding in portfolio.items():
                        if holding['units'] > 0:
                            # Check if asset allocation has drifted significantly
                            current_value = holding['units'] * new_prices[asset]
                            total_value = sum([h['units'] * new_prices[a] for a, h in portfolio.items()])
                            
                            if total_value > 0:
                                current_pct = (current_value / total_value) * 100
                                target_pct = st.session_state.allocator.baseline_allocations[
                                    active_profile.get('profile', 'Moderate')
                                ].get(asset, 0) * 100
                                
                                if abs(current_pct - target_pct) > 10:  # 10% drift
                                    needs_rebalance = True
                                    break
                    
                    if needs_rebalance:
                        st.warning("⚖️ Portfolio may need rebalancing due to price changes!")
                        st.info("💡 Consider updating ML models to rebalance allocations")
                
                st.rerun()
        
        current_prices = st.session_state.market.get_all_prices()
        price_cols = st.columns(4)
        
        with price_cols[0]:
            st.metric(
                "📈 Equity (NIFTY 50)", 
                f"₹{current_prices['equity']:.2f}",
                help="Live NIFTY 50 Index"
            )
        with price_cols[1]:
            st.metric(
                "🥇 Gold (per 10g)", 
                f"₹{current_prices['gold']:.2f}",
                help="Live Gold Futures (INR)"
            )
        with price_cols[2]:
            st.metric(
                "🏦 Fixed Deposit", 
                f"₹{current_prices['fd']:.2f}",
                help="Simulated FD NAV (6.5% annual)"
            )
        with price_cols[3]:
            st.metric(
                "💧 Liquid Fund", 
                f"₹{current_prices['liquid']:.2f}",
                help="Simulated Liquid NAV (4% annual)"
            )
        
        last_update = st.session_state.last_price_update
        st.caption(f"🕐 Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.divider()
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard", 
            "➕ Add Transaction", 
            "💼 Portfolio", 
            "📈 Performance",
            "🔍 XAI & Comparison",
            "⚙️ Settings"
        ])
        
        # TAB 1: Dashboard
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            wallet = db.wallets[user_id]
            portfolio_data = st.session_state.portfolio_sim.calculate_portfolio_value(
                db, user_id
            )
            
            with col1:
                st.metric(
                    "💰 Wallet Balance",
                    f"₹{wallet['balance']:.2f}"
                )
                st.caption(f"Total Rounded Up: ₹{wallet['total_rounded_up']:.2f}")
            
            with col2:
                st.metric(
                    "📊 Total Invested",
                    f"₹{wallet['total_invested']:.2f}"
                )
                st.caption(f"Current Value: ₹{portfolio_data['total_value']:.2f}")
            
            with col3:
                profit_color = "normal" if portfolio_data['profit_loss'] >= 0 else "inverse"
                st.metric(
                    "💵 Profit/Loss",
                    f"₹{portfolio_data['profit_loss']:.2f}",
                    delta=f"{portfolio_data['profit_loss_pct']:.2f}%",
                    delta_color=profit_color
                )
            
            st.divider()
            
            # Portfolio breakdown
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Current Portfolio Allocation")
                
                breakdown = portfolio_data['asset_breakdown']
                if sum(breakdown.values()) > 0:
                    fig = px.pie(
                        values=list(breakdown.values()),
                        names=[name.capitalize() for name in breakdown.keys()],
                        title="Asset Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No investments yet. Add transactions to get started!")
            
            with col2:
                st.subheader("💼 Asset Values")
                
                if sum(portfolio_data['asset_values'].values()) > 0:
                    asset_df = pd.DataFrame([
                        {
                            'Asset': asset.capitalize(),
                            'Value': f"₹{value:.2f}",
                            'Percentage': f"{breakdown[asset]:.1f}%"
                        }
                        for asset, value in portfolio_data['asset_values'].items()
                        if value > 0
                    ])
                    st.dataframe(asset_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No holdings yet")
            
            # Portfolio growth chart
            st.subheader("📈 Portfolio Growth (Last 10 Investments)")
            
            # Get user's investments
            user_investments = [inv for inv in db.investments if inv['user_id'] == user_id]
            
            if len(user_investments) > 0:
                # Get last 10 investments
                recent_investments = user_investments[-10:]
                
                # Create timeline from first recent investment to now
                start_date = recent_investments[0]['timestamp']
                end_date = datetime.now()
                
                dates = []
                values = []
                
                # Sample daily from first investment to now
                current_date = start_date
                while current_date <= end_date:
                    portfolio_value = st.session_state.portfolio_sim.calculate_portfolio_value(
                        db, user_id, current_date
                    )
                    dates.append(current_date)
                    values.append(portfolio_value['total_value'])
                    current_date += timedelta(days=1)
                
                if len(dates) > 0:
                    history_df = pd.DataFrame({'date': dates, 'value': values})
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=history_df['date'],
                        y=history_df['value'],
                        mode='lines+markers',
                        name='Portfolio Value',
                        line=dict(color='#1f77b4', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(31, 119, 180, 0.2)',
                        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Value:</b> ₹%{y:.2f}<extra></extra>'
                    ))
                    
                    # Add investment markers
                    inv_dates = [inv['timestamp'] for inv in recent_investments]
                    inv_values = []
                    for inv_date in inv_dates:
                        pv = st.session_state.portfolio_sim.calculate_portfolio_value(db, user_id, inv_date)
                        inv_values.append(pv['total_value'])
                    
                    fig.add_trace(go.Scatter(
                        x=inv_dates,
                        y=inv_values,
                        mode='markers',
                        name='Investment Points',
                        marker=dict(color='red', size=10, symbol='diamond'),
                        hovertemplate='<b>Investment</b><br>Date: %{x|%Y-%m-%d}<br>Value: ₹%{y:.2f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Value (₹)",
                        hovermode='x unified',
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Starting Value", f"₹{values[0]:.2f}")
                    with col2:
                        st.metric("Current Value", f"₹{values[-1]:.2f}")
                    with col3:
                        growth = ((values[-1] - values[0]) / values[0] * 100) if values[0] > 0 else 0
                        st.metric("Growth", f"{growth:.2f}%", delta=f"₹{values[-1] - values[0]:.2f}")
                else:
                    st.info("Not enough data points yet")
            else:
                st.info("💡 No investment data yet. Add transactions to accumulate wallet balance, then investments will be made automatically when threshold is reached!")
        
        # TAB 2: Add Transaction
        with tab2:
            st.subheader("➕ Add New Transaction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input(
                    "Transaction Amount (₹)",
                    min_value=1.0,
                    max_value=100000.0,
                    value=100.0,
                    step=10.0
                )
                
                category = st.selectbox(
                    "Category",
                    ["grocery_pos", "gas_transport", "misc_net", "shopping_net", 
                     "entertainment", "food_dining", "personal_care", "health_fitness"]
                )
            
            with col2:
                merchant = st.text_input(
                    "Merchant Name",
                    value="Sample Merchant"
                )
                
                round_up_rule = user_info['round_up_rule']
                rounded = np.ceil(amount / round_up_rule) * round_up_rule
                spare = rounded - amount
                
                st.info(f"Round-up to: ₹{rounded:.2f}")
                st.success(f"Spare change: ₹{spare:.2f}")
            
            if st.button("💳 Add Transaction", type="primary"):
                add_new_transaction(user_id, amount, merchant, category)
                st.rerun()
        
        # TAB 3: Portfolio Details
        with tab3:
            st.subheader("💼 Portfolio Holdings")
            
            portfolio = db.get_portfolio(user_id)
            current_prices = st.session_state.market.get_all_prices()
            
            holdings_data = []
            for asset, holding in portfolio.items():
                if holding['units'] > 0:
                    current_price = current_prices[asset]
                    current_value = holding['units'] * current_price
                    profit_loss = current_value - holding['invested']
                    profit_loss_pct = (profit_loss / holding['invested'] * 100) if holding['invested'] > 0 else 0
                    
                    holdings_data.append({
                        'Asset': asset.capitalize(),
                        'Units': f"{holding['units']:.4f}",
                        'Invested (₹)': f"₹{holding['invested']:.2f}",
                        'Current Value (₹)': f"₹{current_value:.2f}",
                        'P/L (₹)': f"₹{profit_loss:.2f}",
                        'P/L (%)': f"{profit_loss_pct:.2f}%"
                    })
            
            if holdings_data:
                holdings_df = pd.DataFrame(holdings_data)
                st.dataframe(holdings_df, use_container_width=True, hide_index=True)
            else:
                st.info("💡 No holdings yet. Add transactions until wallet reaches ₹100 threshold to trigger first investment!")
            
            st.divider()
            
            # Investment history
            st.subheader("📜 Investment History")
            user_investments = [inv for inv in db.investments if inv['user_id'] == user_id]
            
            if user_investments:
                inv_data = []
                for inv in user_investments[-10:]:
                    inv_data.append({
                        'Date': inv['timestamp'].strftime('%Y-%m-%d %H:%M'),
                        'Equity': f"₹{inv['allocation'].get('equity', 0):.2f}",
                        'Gold': f"₹{inv['allocation'].get('gold', 0):.2f}",
                        'FD': f"₹{inv['allocation'].get('fd', 0):.2f}",
                        'Liquid': f"₹{inv['allocation'].get('liquid', 0):.2f}",
                    })
                
                inv_df = pd.DataFrame(inv_data)
                st.dataframe(inv_df, use_container_width=True, hide_index=True)
            else:
                st.info("No investments made yet")
        
        # TAB 4: Performance
        with tab4:
            st.subheader("📈 Asset Performance")
            
            # Get user investments
            user_investments = [inv for inv in db.investments if inv['user_id'] == user_id]
            
            if user_investments:
                # Calculate performance for each asset based on actual holdings
                portfolio = db.get_portfolio(user_id)
                current_prices = st.session_state.market.get_all_prices()
                
                performance_data = []
                has_performance = False
                
                for asset, holding in portfolio.items():
                    if holding['units'] > 0 and holding['invested'] > 0:
                        # Current value
                        current_value = holding['units'] * current_prices[asset]
                        
                        # Calculate return percentage
                        returns_pct = ((current_value - holding['invested']) / holding['invested']) * 100
                        
                        performance_data.append({
                            'Asset': asset.capitalize(),
                            'Invested': holding['invested'],
                            'Current Value': current_value,
                            'Returns (%)': returns_pct
                        })
                        has_performance = True
                
                if has_performance:
                    perf_df = pd.DataFrame(performance_data)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("#### 📊 Performance Summary")
                        display_df = perf_df.copy()
                        display_df['Invested'] = display_df['Invested'].apply(lambda x: f"₹{x:.2f}")
                        display_df['Current Value'] = display_df['Current Value'].apply(lambda x: f"₹{x:.2f}")
                        display_df['Returns (%)'] = display_df['Returns (%)'].apply(lambda x: f"{x:+.2f}%")
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.markdown("#### 📊 Returns Comparison")
                        fig = px.bar(
                            perf_df,
                            x='Asset',
                            y='Returns (%)',
                            title='Asset Returns Comparison',
                            color='Returns (%)',
                            color_continuous_scale='RdYlGn',
                            color_continuous_midpoint=0,
                            text='Returns (%)'
                        )
                        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    
                    # Overall portfolio performance
                    total_invested = perf_df['Invested'].sum()
                    total_current = perf_df['Current Value'].sum()
                    overall_return = ((total_current - total_invested) / total_invested * 100) if total_invested > 0 else 0
                    
                    st.markdown("#### 💼 Overall Portfolio Performance")
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.metric("Total Invested", f"₹{total_invested:.2f}")
                    with metric_cols[1]:
                        st.metric("Current Value", f"₹{total_current:.2f}")
                    with metric_cols[2]:
                        delta_color = "normal" if overall_return >= 0 else "inverse"
                        st.metric(
                            "Overall Returns", 
                            f"{overall_return:+.2f}%",
                            delta=f"₹{total_current - total_invested:.2f}",
                            delta_color=delta_color
                        )
                else:
                    st.info("💡 No holdings with value yet. Portfolio is building up!")
            else:
                st.info("💡 No performance data available yet. Make your first investment to see asset performance!")
            
            st.divider()
            
            # Allocation explanation
            st.subheader("🎯 Current Allocation Strategy")
            explanation = st.session_state.allocator.get_allocation_explanation(
                user_id, active_profile.get('profile', 'Moderate')
            )
            st.text(explanation)
        
        # TAB 5: XAI & Model Comparison
        with tab5:
            st.subheader("🔍 Explainable AI (XAI) - Why This Profile?")
            
            # Feature importance
            if hasattr(st.session_state.profiler_kmeans, 'feature_importance'):
                st.markdown("### 📊 Feature Importance")
                st.markdown("*Which spending patterns matter most for profiling?*")
                
                importance_data = st.session_state.profiler_kmeans.get_feature_importance()
                importance_df = pd.DataFrame([
                    {'Feature': k.replace('_', ' ').title(), 'Importance': v}
                    for k, v in sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
                ])
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance for Risk Profiling',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # User-specific explanation
            st.markdown("### 🎯 Your Profile Explanation")
            
            if hasattr(db, 'user_feature_matrices') and user_id in db.user_feature_matrices:
                feature_vector = db.user_feature_matrices[user_id]
                explanation = st.session_state.profiler_kmeans.explain_user_profile(
                    user_id, feature_vector
                )
                st.markdown(explanation)
            else:
                st.info("Explanation not available")
            
            st.divider()
            
            # Model Comparison
            st.markdown("### 🤖 Model Comparison: K-Means vs Gaussian Mixture")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### K-Means Clustering")
                st.markdown(f"**Profile:** {kmeans_profile.get('profile', 'N/A')}")
                st.markdown(f"**Cluster:** C{kmeans_profile.get('cluster', 0)}")
                st.markdown(f"**Risk Score:** {kmeans_profile.get('risk_score', 0):.3f}")
                
                # K-Means metrics
                if hasattr(st.session_state.profiler_kmeans, 'comparison_metrics'):
                    metrics = st.session_state.profiler_kmeans.comparison_metrics.get('kmeans', {})
                    st.markdown("**Model Metrics:**")
                    st.markdown(f"- Silhouette Score: {metrics.get('silhouette', 0):.3f}")
                    st.markdown(f"- Davies-Bouldin Index: {metrics.get('davies_bouldin', 0):.3f}")
            
            with col2:
                st.markdown("#### Gaussian Mixture Model")
                st.markdown(f"**Profile:** {gmm_profile.get('profile', 'N/A')}")
                st.markdown(f"**Cluster:** C{gmm_profile.get('cluster', 0)}")
                st.markdown(f"**Risk Score:** {gmm_profile.get('risk_score', 0):.3f}")
                
                # GMM-specific: probability distribution
                if 'cluster_probabilities' in gmm_profile:
                    st.markdown("**Cluster Probabilities:**")
                    probs = gmm_profile['cluster_probabilities']
                    for i, p in enumerate(probs):
                        st.markdown(f"- Cluster {i}: {p:.1%}")
                
                # GMM metrics
                if hasattr(st.session_state.profiler_gmm, 'comparison_metrics'):
                    metrics = st.session_state.profiler_gmm.comparison_metrics.get('gmm', {})
                    st.markdown("**Model Metrics:**")
                    st.markdown(f"- Silhouette Score: {metrics.get('silhouette', 0):.3f}")
                    st.markdown(f"- BIC: {metrics.get('bic', 0):.1f}")
            
            st.divider()
            
            # Portfolio growth comparison
            st.markdown("### 📈 Portfolio Growth Comparison (Simulated)")
            st.markdown("*If we had used each model from the start*")
            
            # Simulate what portfolio would look like with each model
            user_investments = [inv for inv in db.investments if inv['user_id'] == user_id]
            
            if user_investments:
                st.info(f"✅ With K-Means: {kmeans_profile.get('profile', 'Moderate')} allocation")
                st.info(f"✅ With GMM: {gmm_profile.get('profile', 'Moderate')} allocation")
                
                # Show allocation differences
                km_alloc = st.session_state.allocator.baseline_allocations.get(
                    kmeans_profile.get('profile', 'Moderate'), {}
                )
                gmm_alloc = st.session_state.allocator.baseline_allocations.get(
                    gmm_profile.get('profile', 'Moderate'), {}
                )
                
                comparison_df = pd.DataFrame({
                    'Asset': ['Equity', 'Gold', 'FD', 'Liquid'],
                    'K-Means %': [
                        km_alloc.get('equity', 0) * 100,
                        km_alloc.get('gold', 0) * 100,
                        km_alloc.get('fd', 0) * 100,
                        km_alloc.get('liquid', 0) * 100
                    ],
                    'GMM %': [
                        gmm_alloc.get('equity', 0) * 100,
                        gmm_alloc.get('gold', 0) * 100,
                        gmm_alloc.get('fd', 0) * 100,
                        gmm_alloc.get('liquid', 0) * 100
                    ]
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='K-Means',
                    x=comparison_df['Asset'],
                    y=comparison_df['K-Means %'],
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    name='GMM',
                    x=comparison_df['Asset'],
                    y=comparison_df['GMM %'],
                    marker_color='lightcoral'
                ))
                fig.update_layout(
                    title='Allocation Comparison',
                    yaxis_title='Percentage (%)',
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("💡 Make your first investment to see model comparison!")
        
        # TAB 6: Settings
        with tab6:
            st.subheader("⚙️ User Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_round_up = st.selectbox(
                    "Round-up Rule",
                    [10, 20, 50, 100],
                    index=[10, 20, 50, 100].index(user_info['round_up_rule'])
                )
                
                if st.button("Update Round-up Rule"):
                    db.users[user_id]['round_up_rule'] = new_round_up
                    st.success(f"✅ Round-up rule updated to ₹{new_round_up}")
            
            with col2:
                new_threshold = st.number_input(
                    "Investment Threshold (₹)",
                    min_value=50.0,
                    max_value=1000.0,
                    value=float(user_info['threshold']),
                    step=50.0
                )
                
                if st.button("Update Threshold"):
                    db.users[user_id]['threshold'] = new_threshold
                    st.success(f"✅ Threshold updated to ₹{new_threshold}")
            
            st.divider()
            
            st.subheader("🎲 Quick Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💼 Trigger Batch Investment"):
                    execute_batch_investment(user_id)
            
            with col2:
                if st.button("🔄 Reset Wallet"):
                    db.wallets[user_id]['balance'] = 0.0
                    st.success("✅ Wallet reset!")
                    st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💡 Smart Investment Round-Up System | ML-Powered Portfolio Management</p>
    <p style='font-size: 0.8em;'>Round up spare change → Auto-invest → Grow wealth 📈</p>
    <p style='font-size: 0.8em;'>🤖 K-Means vs GMM | 🔍 XAI Explainability | 📊 Real-time Analytics</p>
</div>
""", unsafe_allow_html=True)