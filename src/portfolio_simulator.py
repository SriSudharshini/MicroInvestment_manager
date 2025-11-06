import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

class MarketDataSimulator:
    """
    Market data with REAL-TIME prices from Yahoo Finance
    """
    
    def __init__(self, use_real_data=True):
        self.use_real_data = use_real_data
        
        # Asset ticker mapping to Yahoo Finance symbols
        self.tickers = {
            'equity': '^NSEI',      # NIFTY 50 Index
            'gold': 'GC=F',         # Gold Futures
            'fd': None,             # Fixed Deposit (simulated)
            'liquid': None          # Liquid funds (simulated)
        }
        
        # Base prices (fallback if API fails)
        self.base_prices = {
            'equity': 19500.0,   # NIFTY 50 approximate
            'gold': 62000.0,     # Gold per 10g (INR)
            'fd': 100.0,         # Fixed Deposit unit
            'liquid': 100.0      # Liquid fund NAV
        }
        
        self.current_prices = self.base_prices.copy()
        
        # Annual returns (for simulation)
        self.expected_returns = {
            'equity': 0.12,
            'gold': 0.08,
            'fd': 0.065,
            'liquid': 0.04
        }
        
        # Volatility
        self.volatility = {
            'equity': 0.18,
            'gold': 0.12,
            'fd': 0.01,
            'liquid': 0.005
        }
        
        self.start_date = datetime(2023, 1, 1)
        self.historical_data = {}
        
        # Initialize with real data if available
        if self.use_real_data:
            self._fetch_real_prices()
        
        # Generate synthetic history for backtesting
        self._generate_synthetic_history()
    
    def _fetch_real_prices(self):
        """Fetch current prices from Yahoo Finance"""
        try:
            import yfinance as yf
            
            # Fetch NIFTY 50
            try:
                nifty = yf.Ticker('^NSEI')
                nifty_data = nifty.history(period='1d')
                if not nifty_data.empty:
                    self.current_prices['equity'] = float(nifty_data['Close'].iloc[-1])
            except:
                pass
            
            # Fetch Gold
            try:
                gold = yf.Ticker('GC=F')
                gold_data = gold.history(period='1d')
                if not gold_data.empty:
                    # Convert USD per troy ounce to INR per 10g
                    gold_usd = float(gold_data['Close'].iloc[-1])
                    usd_to_inr = 83.0  # Approximate exchange rate
                    troy_ounce_to_10g = 0.3215  # 1 troy ounce = 31.1g
                    self.current_prices['gold'] = gold_usd * usd_to_inr * troy_ounce_to_10g
            except:
                pass
            
            # FD and Liquid grow steadily (simulated)
            days_since_start = (datetime.now() - self.start_date).days
            daily_fd_return = self.expected_returns['fd'] / 365
            daily_liquid_return = self.expected_returns['liquid'] / 365
            
            self.current_prices['fd'] = self.base_prices['fd'] * (1 + daily_fd_return) ** days_since_start
            self.current_prices['liquid'] = self.base_prices['liquid'] * (1 + daily_liquid_return) ** days_since_start
            
        except Exception as e:
            print(f"Error fetching real prices: {e}")
            # Keep base prices as fallback
    
    def _generate_synthetic_history(self, days=730):
        """Generate synthetic price history using geometric Brownian motion"""
        dates = [self.start_date + timedelta(days=i) for i in range(days)]
        
        for asset in ['equity', 'gold', 'fd', 'liquid']:
            prices = [self.current_prices[asset]]
            
            daily_return = self.expected_returns[asset] / 252  # 252 trading days
            daily_vol = self.volatility[asset] / np.sqrt(252)
            
            for _ in range(days - 1):
                random_shock = np.random.normal(0, daily_vol)
                price_change = prices[-1] * (daily_return + random_shock)
                new_price = max(prices[-1] + price_change, prices[-1] * 0.5)  # Floor at 50% of previous
                prices.append(new_price)
            
            self.historical_data[asset] = pd.DataFrame({
                'date': dates,
                'price': prices
            })
            
            # Update current price to latest
            self.current_prices[asset] = prices[-1]
    
    def get_price(self, asset, date=None):
        """Get price for an asset on a specific date"""
        if date is None:
            # Return current live price
            if self.use_real_data:
                self._fetch_real_prices()  # Refresh prices
            return self.current_prices[asset]
        
        # Historical price lookup
        if asset not in self.historical_data:
            return self.current_prices[asset]
        
        df = self.historical_data[asset]
        
        # Find closest date
        closest_idx = (df['date'] - date).abs().argsort()[0]
        return df.iloc[closest_idx]['price']
    
    def get_all_prices(self, date=None):
        """Get prices for all assets"""
        if date is None and self.use_real_data:
            self._fetch_real_prices()  # Refresh for current prices
        
        return {asset: self.get_price(asset, date) for asset in self.current_prices.keys()}
    
    def calculate_returns(self, asset, start_date, end_date):
        """Calculate returns for an asset between two dates"""
        start_price = self.get_price(asset, start_date)
        end_price = self.get_price(asset, end_date)
        
        return ((end_price - start_price) / start_price) * 100
    
    def get_performance_data(self, days_back=30):
        """Get recent performance data for all assets"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        performance = {}
        for asset in self.current_prices.keys():
            returns = self.calculate_returns(asset, start_date, end_date)
            performance[asset] = returns
        
        return performance


class PortfolioSimulator:
    """
    Simulates portfolio investments and tracks performance
    """
    
    def __init__(self, market_simulator):
        self.market = market_simulator
        self.transaction_fee = 0.01  # 1% transaction fee
        self.tax_rate = 0.10  # 10% tax on gains
    
    def execute_investment(self, db, user_id, allocation, timestamp=None):
        """
        Execute an investment with the given allocation
        
        Args:
            db: Database instance
            user_id: User ID
            allocation: Dictionary with asset: amount
            timestamp: Investment timestamp
        
        Returns:
            Investment details
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get current prices
        prices = self.market.get_all_prices(timestamp)
        
        # Calculate total investment amount
        total_amount = sum(allocation.values())
        
        # Deduct fees
        fees = total_amount * self.transaction_fee
        net_investment = total_amount - fees
        
        # Adjust allocation for fees (proportionally reduce each)
        adjusted_allocation = {}
        for asset, amount in allocation.items():
            adjusted_allocation[asset] = amount * (net_investment / total_amount)
        
        # Check if wallet has sufficient balance
        if not db.deduct_from_wallet(user_id, total_amount):
            return None
        
        # Record investment in database
        investment = db.add_investment(user_id, adjusted_allocation, prices, timestamp)
        
        return {
            'investment_id': investment['inv_id'],
            'amount': total_amount,
            'fees': fees,
            'net_investment': net_investment,
            'allocation': adjusted_allocation,
            'prices': prices,
            'timestamp': timestamp
        }
    
    def calculate_portfolio_value(self, db, user_id, date=None):
        """
        Calculate current portfolio value for a user
        
        Args:
            db: Database instance
            user_id: User ID
            date: Valuation date (None for current)
        
        Returns:
            Dictionary with portfolio details
        """
        portfolio = db.get_portfolio(user_id)
        current_prices = self.market.get_all_prices(date)
        
        total_value, asset_values = db.get_portfolio_value(user_id, current_prices)
        total_invested = db.wallets[user_id]['total_invested']
        
        # Calculate profit/loss
        profit_loss = total_value - total_invested
        profit_loss_pct = (profit_loss / total_invested * 100) if total_invested > 0 else 0
        
        # Calculate unrealized gains (before tax)
        unrealized_gains = profit_loss
        tax_liability = max(unrealized_gains, 0) * self.tax_rate
        net_value = total_value - tax_liability
        
        return {
            'total_value': total_value,
            'net_value': net_value,
            'total_invested': total_invested,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'tax_liability': tax_liability,
            'asset_values': asset_values,
            'asset_breakdown': self._calculate_breakdown(asset_values)
        }
    
    def _calculate_breakdown(self, asset_values):
        """Calculate percentage breakdown of assets"""
        total = sum(asset_values.values())
        if total == 0:
            return {asset: 0 for asset in asset_values}
        
        return {asset: (value / total * 100) for asset, value in asset_values.items()}
    
    def get_portfolio_history(self, db, user_id, days=30):
        """
        Get historical portfolio values
        
        Args:
            db: Database instance
            user_id: User ID
            days: Number of days to look back
        
        Returns:
            DataFrame with date and portfolio value
        """
        # Get user's first investment date
        user_investments = [inv for inv in db.investments if inv['user_id'] == user_id]
        
        if not user_investments:
            # No investments yet, return empty dataframe
            return pd.DataFrame({'date': [], 'value': []})
        
        # Start from first investment date
        first_investment_date = min(inv['timestamp'] for inv in user_investments)
        end_date = datetime.now()
        start_date = max(first_investment_date, end_date - timedelta(days=days))
        
        dates = []
        values = []
        
        # Sample every day
        current_date = start_date
        while current_date <= end_date:
            portfolio_value = self.calculate_portfolio_value(db, user_id, current_date)
            dates.append(current_date)
            values.append(portfolio_value['total_value'])
            current_date += timedelta(days=1)
        
        return pd.DataFrame({
            'date': dates,
            'value': values
        })
    
    def get_asset_performance(self, db, user_id, days=30):
        """Get performance of each asset in portfolio"""
        portfolio = db.get_portfolio(user_id)
        performance = {}
        
        # Get user's investments
        user_investments = [inv for inv in db.investments if inv['user_id'] == user_id]
        
        if not user_investments:
            return performance
        
        for asset, holding in portfolio.items():
            if holding['units'] > 0:
                returns = self.market.calculate_returns(
                    asset,
                    datetime.now() - timedelta(days=days),
                    datetime.now()
                )
                performance[asset] = returns
        
        return performance