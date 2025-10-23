# 💰 Smart Investment Round-Up System

An ML-powered investment platform that automatically rounds up transactions and intelligently invests spare change into diversified portfolios.

## 🎯 Features

- **Automatic Round-Up**: Rounds transaction amounts and saves spare change
- **ML-Based User Profiling**: Clusters users into risk profiles (Conservative/Moderate/Aggressive)
- **Smart Allocation**: Rule-based baseline with ML-driven adjustments
- **Portfolio Simulation**: Backtests investments using historical market data
- **Interactive Dashboard**: Real-time portfolio tracking and analytics
- **Performance Tracking**: Monitor returns, profit/loss, and asset performance

## 📁 Project Structure

```
investment-roundup/
├── data/
│   ├── raw/
│   │   └── transactions.csv        # Your transaction dataset
│   └── processed/
│       └── database.pkl             # Saved database state
├── models/
│   └── saved_models/
│       └── user_profiler.pkl        # Trained ML model
├── src/
│   ├── __init__.py
│   ├── database.py                  # In-memory database
│   ├── data_preprocessing.py        # Data loading & feature engineering
│   ├── user_profiling.py            # ML clustering model
│   ├── allocation_engine.py         # Investment allocation logic
│   └── portfolio_simulator.py       # Market simulation & portfolio tracking
├── app.py                           # Streamlit UI application
├── requirements.txt
└── README.md
```

## 🚀 Setup Instructions

### Step 1: Create Project Directory

```bash
mkdir investment-roundup
cd investment-roundup
```

### Step 2: Create Subdirectories

```bash
mkdir -p data/raw data/processed models/saved_models src
```

### Step 3: Add Your Data

1. Place your `transactions.csv` file in the `data/raw/` folder
2. The CSV should have columns: `trans_date_trans_time`, `cc_num`, `merchant`, `category`, `amt`, `first`, `last`, `gender`, `city`, `state`, `job`, `dob`

### Step 4: Create Empty `__init__.py`

```bash
touch src/__init__.py
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 How to Use

### Initial Setup

1. **Click "Initialize System"** in the sidebar
   - Loads 5-10 sample users from your dataset
   - Builds ML profiling model using transaction history
   - Creates initial portfolio allocations

2. **Select a User** from the dropdown
   - View their risk profile and cluster assignment
   - See current wallet balance and portfolio

### Dashboard Tab

- **Wallet Balance**: Current spare change accumulated
- **Total Invested**: Amount invested in portfolio
- **Profit/Loss**: Current returns with percentage
- **Portfolio Pie Chart**: Visual breakdown of asset allocation
- **Growth Chart**: 30-day portfolio value history

### Add Transaction Tab

1. Enter transaction amount (e.g., ₹175)
2. Select category (grocery, entertainment, etc.)
3. Enter merchant name
4. Click "Add Transaction"
5. System automatically:
   - Rounds up to nearest ₹10/20/50 (configurable)
   - Adds spare change to wallet
   - Triggers batch investment when threshold reached

### Portfolio Tab

- **Holdings**: Detailed view of all assets
- **Investment History**: Record of all batch investments
- **Units Owned**: Quantity of each asset
- **Current Value**: Real-time valuation

### Performance Tab

- **Asset Returns**: 30-day performance of each asset
- **Comparison Charts**: Visual performance comparison
- **Allocation Strategy**: Explanation of current ML-driven allocation

### Settings Tab

- **Round-up Rule**: Change rounding amount (₹10/20/50/100)
- **Investment Threshold**: Set minimum wallet balance to trigger investment
- **Quick Actions**: Manual investment trigger, wallet reset

## 🧠 ML Model Explained

### User Profiling (Clustering)

**Algorithm**: K-Means Clustering (3 clusters)

**Features Used**:
1. Average monthly spending
2. Standard deviation of spending (volatility)
3. Average transaction amount
4. Transaction frequency
5. Number of unique merchants
6. Category diversity

**Output**: 
- Conservative (Low spend, low volatility)
- Moderate (Medium spend, medium volatility)
- Aggressive (High spend, high volatility)

**Purpose**: Determines initial risk tolerance and investment allocation

### Allocation Engine

**Phase 1: Rule-Based Baseline**

```
Conservative: 10% Equity, 30% Gold, 50% FD, 10% Liquid
Moderate:     40% Equity, 25% Gold, 25% FD, 10% Liquid
Aggressive:   65% Equity, 20% Gold, 10% FD, 5% Liquid
```

**Phase 2: ML-Driven Adjustments**

- Monitors 30-day asset performance
- Calculates relative performance vs. average
- Adjusts weights using learning rate (3%)
- Applies safety constraints:
  - Max equity: 80%
  - Min liquid: 5%
  - Max shift per update: 10%

**Formula**:
```
new_weight = old_weight × (1 + learning_rate × performance_delta)
```

**Example**: If equity returns 8% and average is 5%:
```
adjustment = 0.03 × (8 - 5) / 100 = 0.0009
new_equity_weight = 0.40 × (1 + 0.0009) = 0.4036 (40.36%)
```

## 📈 Market Simulation

The system uses synthetic market data generated via **Geometric Brownian Motion**:

- **Equity**: 12% annual return, 18% volatility
- **Gold**: 8% annual return, 12% volatility
- **FD (Fixed Deposit)**: 6.5% annual return, 1% volatility
- **Liquid**: 4% annual return, 0.5% volatility

**Transaction Costs**: 1% fee on all investments  
**Tax**: 10% on realized gains

## 🎮 Testing Scenarios

### Scenario 1: Conservative User Journey

1. Add 5 small transactions (₹50-₹200)
2. Watch wallet accumulate to threshold
3. System invests with conservative allocation
4. View predominantly FD and Gold holdings
5. Steady, low-risk returns

### Scenario 2: Aggressive User Journey

1. Add 3 large transactions (₹500-₹2000)
2. Quick threshold trigger
3. High equity allocation (65%+)
4. Higher potential returns (and volatility)
5. ML adjusts based on equity performance

### Scenario 3: ML Learning

1. Run for 30 days with regular transactions
2. Click "Update ML Models" in sidebar
3. System analyzes asset performance
4. Weights shift toward better performers
5. View updated allocation in Performance tab

## 🔧 Configuration Options

### Round-Up Rules

- **₹10**: Frequent small round-ups
- **₹20**: Moderate accumulation
- **₹50**: Standard (recommended)
- **₹100**: Larger spare change amounts

### Investment Thresholds

- **₹50**: Quick, frequent investments
- **₹100**: Balanced approach (recommended)
- **₹500**: Larger, less frequent batches
- **₹1000**: Maximum accumulation

## 📊 Key Metrics Explained

| Metric | Description |
|--------|-------------|
| **Wallet Balance** | Unspent spare change ready to invest |
| **Total Rounded Up** | Cumulative spare change collected |
| **Total Invested** | Amount deployed in portfolio |
| **Current Value** | Real-time portfolio worth |
| **Profit/Loss** | Returns after fees, before tax |
| **Net Value** | Portfolio value after estimated tax |
| **Risk Score** | 0-1 scale of user's risk tolerance |

## 🎯 Business Logic Flow

```
1. Transaction occurs (₹175 at café)
   ↓
2. Round up to ₹200 (rule: nearest ₹50)
   ↓
3. Spare change: ₹25 → Add to wallet
   ↓
4. Check wallet balance vs threshold
   ↓
5. If threshold met → Trigger investment
   ↓
6. Get user profile (Moderate)
   ↓
7. Calculate allocation (40% Equity, 25% Gold, 25% FD, 10% Liquid)
   ↓
8. Apply ML adjustments (if learned)
   ↓
9. Execute trades at current market prices
   ↓
10. Deduct 1% transaction fee
   ↓
11. Update portfolio holdings
   ↓
12. Display on dashboard
```

## 🧪 ML Training Process

### Offline Training (Initialization)

```python
# Extract features from transaction history
features = compute_spending_features(transactions_df, user_id)

# Build feature matrix for all users
feature_matrix = [prepare_ml_features(f) for f in all_features]

# Train clustering model
profiler = UserProfiler(n_clusters=3)
user_profiles = profiler.fit(feature_matrix, user_ids)

# Assign profiles to users
for user_id, profile in user_profiles.items():
    db.set_user_profile(user_id, profile['profile'], profile['risk_score'])
```

### Online Learning (Updates)

```python
# Every 30 days, evaluate performance
performance = portfolio_sim.get_asset_performance(db, user_id, days=30)

# Update allocation weights
new_weights = allocator.update_weights(user_id, profile, performance)

# Next investment uses adjusted weights
```

## 🛡️ Safety Features

1. **Minimum Liquid Buffer**: Always keeps 5% in liquid assets
2. **Maximum Equity Cap**: Prevents over-exposure (80% max)
3. **Gradual Adjustments**: ML changes weights slowly (3% learning rate)
4. **Transaction Fees**: Realistic 1% cost simulation
5. **Threshold Batching**: Reduces micro-transactions

## 📦 Files You Need to Create

### 1. `requirements.txt` (Already provided in artifacts)

### 2. `src/__init__.py` (Empty file)

```bash
touch src/__init__.py
```

### 3. `src/database.py` (Already provided in artifacts)

### 4. `src/data_preprocessing.py` (Already provided in artifacts)

### 5. `src/user_profiling.py` (Already provided in artifacts)

### 6. `src/allocation_engine.py` (Already provided in artifacts)

### 7. `src/portfolio_simulator.py` (Already provided in artifacts)

### 8. `app.py` (Already provided in artifacts)

## 🚀 Quick Start Command Sequence

```bash
# Create project
mkdir investment-roundup && cd investment-roundup

# Create structure
mkdir -p data/raw data/processed models/saved_models src

# Create files (copy from artifacts above)
touch src/__init__.py
# Copy requirements.txt
# Copy all src/*.py files
# Copy app.py

# Place your data
cp /path/to/your/transactions.csv data/raw/

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## 🎓 Educational Value

This project demonstrates:

1. **Real-world ML application** with explainable results
2. **Feature engineering** from transaction data
3. **Unsupervised learning** (K-Means clustering)
4. **Online learning** with performance feedback
5. **Financial simulation** with realistic constraints
6. **Interactive visualization** with Streamlit
7. **Modular architecture** with clean separation of concerns

## 🔍 Troubleshooting

### Issue: "No module named 'src'"

**Solution**: Ensure `src/__init__.py` exists and you're running from project root

### Issue: "transactions.csv not found"

**Solution**: Place CSV in `data/raw/` folder with exact filename

### Issue: "Not enough data for clustering"

**Solution**: Ensure at least 5-10 users with 10+ transactions each

### Issue: Portfolio value shows ₹0

**Solution**: Add transactions until wallet reaches threshold, or manually trigger investment

### Issue: ML model not updating

**Solution**: Click "Update ML Models" in sidebar after 30+ days of data

## 📝 Next Steps / Enhancements

1. **Real bank integration** (via APIs)
2. **Goal-based investing** (save for house, car, etc.)
3. **Social features** (compare with friends)
4. **Advanced ML models** (Reinforcement Learning for allocation)
5. **Real-time market data** (via yfinance or APIs)
6. **Mobile app** (React Native)
7. **Automated rebalancing** (quarterly portfolio adjustments)
8. **Tax optimization** (LTCG vs STCG strategies)

## 📄 License

Educational project - Free to use and modify

## 👨‍💻 Author

Built as an ML engineering portfolio project demonstrating end-to-end system design, from data preprocessing to interactive deployment.

---

**Happy Investing! 💰📈**



