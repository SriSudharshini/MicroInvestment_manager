import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

class MarketDataSimulator:
    """
    Simulates market data using historical prices or synthetic data
    """
    
    def __init__(self):
        self.historical_data = {}
        self.current_prices = {
            'equity': 100.0,  # Base price
            'gold': 50000.0,  # Per 10g
            'fd': 100.0,      # Base (grows by interest rate)
            'liquid': 100.0   # Stable
        }
        
        # Annual returns (approximate)
        self.expected_returns = {
            'equity': 0.12,    # 12% annual
            'gold': 0.08,      # 8% annual
            'fd': 0.065,       # 6.5% annual
            'liquid': 0.04     # 4% annual
        }
        
        # Volatility (standard deviation)
        self.volatility = {
            'equity': 0.18,
            'gold': 0.12,
            'fd': 0.01,
            'liquid': 0.005
        }
        
        self.start_date = datetime(2023, 1, 1)
        self._generate_synthetic_history()
    
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
            return self.current_prices[asset]
        
        if asset not in self.historical_data:
            return self.current_prices[asset]
        
        df = self.historical_data[asset]
        closest_date = df.iloc[(df['date'] - date).abs().argsort()[:1]]
        return closest_date['price'].values[0]
    
    def get_all_prices(self, date=None):
        """Get prices for all assets"""
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
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
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
        
        for asset, holding in portfolio.items():
            if holding['units'] > 0:
                returns = self.market.calculate_returns(
                    asset,
                    datetime.now() - timedelta(days=days),
                    datetime.now()
                )
                performance[asset] = returns
        
        return performance