import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database import InvestmentDatabase
from data_preprocessing import (
    load_and_preprocess_data, 
    compute_spending_features,
    prepare_ml_features,
    create_sample_users_dataset
)
from user_profiling import UserProfiler, build_user_profiles
from allocation_engine import AllocationEngine, check_batch_trigger
from portfolio_simulator import MarketDataSimulator, PortfolioSimulator

# Page config
st.set_page_config(
    page_title="Smart Investment Round-Up",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.db = InvestmentDatabase()
    st.session_state.market = MarketDataSimulator()
    st.session_state.portfolio_sim = PortfolioSimulator(st.session_state.market)
    st.session_state.allocator = AllocationEngine()
    st.session_state.profiler = UserProfiler()
    st.session_state.transactions_df = None
    st.session_state.selected_user = None

def initialize_system():
    """Initialize the system with sample data"""
    with st.spinner("Loading and processing data..."):
        # Load transaction data
        if os.path.exists('data/raw/transactions.csv'):
            df = load_and_preprocess_data('data/raw/transactions.csv', num_users=10)
            st.session_state.transactions_df = df
            
            # Create users in database
            user_ids = create_sample_users_dataset(df, st.session_state.db)
            
            # Load historical transactions into database (sample - last 30 days per user)
            for user_id in user_ids:
                user_trans = df[df['user_id'] == user_id].tail(20)  # Last 20 transactions
                for _, trans in user_trans.iterrows():
                    st.session_state.db.add_transaction(
                        user_id,
                        trans['amount'],
                        trans['merchant'],
                        trans['category'],
                        trans['timestamp']
                    )
            
            # Build user profiles
            st.session_state.profiler = build_user_profiles(
                df, 
                user_ids, 
                st.session_state.db
            )
            
            st.session_state.initialized = True
            st.success(f"✅ System initialized with {len(user_ids)} users!")
        else:
            st.error("❌ Please place your transactions.csv file in data/raw/ folder")

def add_new_transaction(user_id, amount, merchant, category):
    """Add a new transaction and check for batch investment"""
    # Add transaction
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
        
        # Show allocation
        with st.expander("View Allocation Details"):
            for asset, amount in result['allocation'].items():
                st.write(f"• {asset.capitalize()}: ₹{amount:.2f}")
    else:
        st.error("Investment failed - insufficient wallet balance")

def update_ml_allocations():
    """Update ML-based allocations for all users"""
    db = st.session_state.db
    
    for user_id in db.users.keys():
        # Get recent performance
        performance = st.session_state.portfolio_sim.get_asset_performance(
            db, user_id, days=30
        )
        
        if performance:
            profile = db.users[user_id]['profile']
            new_weights, adjustments = st.session_state.allocator.update_weights(
                user_id, profile, performance
            )
    
    st.success("✅ ML allocations updated for all users!")

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
        
        if st.button("🔄 Update ML Models"):
            update_ml_allocations()
        
        st.divider()
        
        # User selection
        st.header("👤 Select User")
        user_list = list(st.session_state.db.users.keys())
        user_names = [f"{st.session_state.db.users[uid]['name']} ({uid[:8]}...)" 
                     for uid in user_list]
        
        selected_idx = st.selectbox(
            "Choose User",
            range(len(user_list)),
            format_func=lambda i: user_names[i]
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
    """)
else:
    user_id = st.session_state.selected_user
    db = st.session_state.db
    
    if user_id:
        user_info = db.users[user_id]
        user_profile = db.user_profiles.get(user_id, {'profile': 'Moderate', 'risk_score': 0.5})
        
        # User header
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("User", user_info['name'])
        with col2:
            st.metric("Risk Profile", user_profile['profile'])
        with col3:
            st.metric("Risk Score", f"{user_profile['risk_score']:.2f}")
        with col4:
            cluster = user_profile.get('cluster', 0)
            st.metric("Cluster", f"C{cluster}")
        
        st.divider()
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", 
            "➕ Add Transaction", 
            "💼 Portfolio", 
            "📈 Performance",
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
                    f"₹{wallet['balance']:.2f}",
                    delta=None
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
                
                asset_df = pd.DataFrame([
                    {
                        'Asset': asset.capitalize(),
                        'Value (₹)': f"₹{value:.2f}",
                        'Percentage': f"{breakdown[asset]:.1f}%"
                    }
                    for asset, value in portfolio_data['asset_values'].items()
                ])
                
                if not asset_df.empty:
                    st.dataframe(asset_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No holdings yet")
            
            # Portfolio growth chart
            st.subheader("📈 Portfolio Growth (Last 30 Days)")
            history = st.session_state.portfolio_sim.get_portfolio_history(
                db, user_id, days=30
            )
            
            if len(history) > 0 and history['value'].sum() > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history['date'],
                    y=history['value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#1f77b4', width=2),
                    fill='tozeroy'
                ))
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Value (₹)",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to show growth chart yet")
        
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
                st.info("No holdings yet. Make your first investment!")
            
            st.divider()
            
            # Investment history
            st.subheader("📜 Investment History")
            user_investments = [inv for inv in db.investments if inv['user_id'] == user_id]
            
            if user_investments:
                inv_data = []
                for inv in user_investments[-10:]:  # Last 10
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
            st.subheader("📈 Asset Performance (Last 30 Days)")
            
            performance = st.session_state.portfolio_sim.get_asset_performance(
                db, user_id, days=30
            )
            
            if performance:
                perf_df = pd.DataFrame([
                    {'Asset': asset.capitalize(), 'Returns (%)': f"{returns:.2f}%"}
                    for asset, returns in performance.items()
                ])
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.dataframe(perf_df, use_container_width=True, hide_index=True)
                
                with col2:
                    fig = px.bar(
                        x=list(performance.keys()),
                        y=list(performance.values()),
                        labels={'x': 'Asset', 'y': 'Returns (%)'},
                        title='Asset Returns Comparison',
                        color=list(performance.values()),
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance data available yet")
            
            st.divider()
            
            # Allocation explanation
            st.subheader("🎯 Current Allocation Strategy")
            explanation = st.session_state.allocator.get_allocation_explanation(
                user_id, user_profile['profile']
            )
            st.text(explanation)
        
        # TAB 5: Settings
        with tab5:
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
</div>
""", unsafe_allow_html=True)