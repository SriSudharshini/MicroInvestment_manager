import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os

class InvestmentDatabase:
    """Simple in-memory database for the investment system"""
    
    def __init__(self):
        self.users = {}
        self.transactions = []
        self.wallets = {}
        self.portfolios = {}
        self.investments = []
        self.user_profiles = {}
        
    # Replace the existing add_user and add_transaction methods in InvestmentDatabase

    def add_user(self, user_id, name, initial_balance=0.0):
        """Add a new user to the system (idempotent). Returns user dict."""
        user_id = str(user_id)
        if user_id not in self.users:
            self.users[user_id] = {
                'user_id': user_id,
                'name': name,
                'created_at': datetime.now(),
                'round_up_rule': 50,  # Round to nearest 50
                'threshold': 100.0,   # Minimum wallet balance to invest
                'profile': 'Moderate'
            }
            self.wallets[user_id] = {
                'balance': float(initial_balance),
                'total_rounded_up': 0.0,
                'total_invested': 0.0
            }
            self.portfolios[user_id] = {
                'equity': {'units': 0.0, 'invested': 0.0},
                'gold': {'units': 0.0, 'invested': 0.0},
                'fd': {'units': 0.0, 'invested': 0.0},
                'liquid': {'units': 0.0, 'invested': 0.0}
            }
        return self.users[user_id]

    def add_transaction(self, user_id, amount, merchant, category, timestamp=None):
        """Add a transaction and calculate round-up. Returns transaction dict.

        If user doesn't exist, raises KeyError.
        """
        user_id = str(user_id)
        if user_id not in self.users:
            raise KeyError(f"User {user_id} not found. Call add_user() first.")

        if timestamp is None:
            timestamp = datetime.now()

        # ensure numeric amount
        try:
            amount_val = float(amount)
        except Exception:
            raise ValueError("Invalid transaction amount")

        round_up_rule = self.users[user_id].get('round_up_rule', 50)
        if round_up_rule <= 0:
            round_up_rule = 50

        # calculate spare change using exact math
        rounded_amount = float(np.ceil(amount_val / round_up_rule) * round_up_rule)
        spare_change = rounded_amount - amount_val
        spare_change = round(float(spare_change), 2)

        trans_id = len(self.transactions)
        transaction = {
            'trans_id': trans_id,
            'user_id': user_id,
            'amount': amount_val,
            'merchant': merchant,
            'category': category,
            'timestamp': timestamp,
            'spare_change': spare_change,
            'rounded_amount': rounded_amount
        }

        self.transactions.append(transaction)

        # Update wallet
        self.wallets[user_id]['balance'] = round(self.wallets[user_id]['balance'] + spare_change, 2)
        self.wallets[user_id]['total_rounded_up'] = round(self.wallets[user_id]['total_rounded_up'] + spare_change, 2)

        return transaction

    
    def get_user_transactions(self, user_id, days=30):
        """Get recent transactions for a user"""
        user_trans = [t for t in self.transactions if t['user_id'] == user_id]
        cutoff = datetime.now() - timedelta(days=days)
        return [t for t in user_trans if t['timestamp'] > cutoff]
    
    def get_wallet_balance(self, user_id):
        """Get current wallet balance"""
        return self.wallets[user_id]['balance']
    
    def deduct_from_wallet(self, user_id, amount):
        """Deduct amount from wallet when investing"""
        if self.wallets[user_id]['balance'] >= amount:
            self.wallets[user_id]['balance'] -= amount
            self.wallets[user_id]['total_invested'] += amount
            return True
        return False
    
    def add_investment(self, user_id, allocation, prices, timestamp=None):
        """Record an investment"""
        if timestamp is None:
            timestamp = datetime.now()
        
        investment = {
            'inv_id': len(self.investments),
            'user_id': user_id,
            'timestamp': timestamp,
            'allocation': allocation.copy(),
            'prices': prices.copy()
        }
        
        # Update portfolio
        for asset, amount in allocation.items():
            if amount > 0 and asset in prices:
                units = amount / prices[asset]
                self.portfolios[user_id][asset]['units'] += units
                self.portfolios[user_id][asset]['invested'] += amount
        
        self.investments.append(investment)
        return investment
    
    def get_portfolio(self, user_id):
        """Get current portfolio holdings"""
        return self.portfolios[user_id]
    
    def get_portfolio_value(self, user_id, current_prices):
        """Calculate current portfolio value"""
        portfolio = self.portfolios[user_id]
        total_value = 0
        asset_values = {}
        
        for asset, holding in portfolio.items():
            if asset in current_prices:
                value = holding['units'] * current_prices[asset]
                asset_values[asset] = value
                total_value += value
        
        return total_value, asset_values
    
    def set_user_profile(self, user_id, profile, risk_score):
        """Set user risk profile"""
        self.user_profiles[user_id] = {
            'profile': profile,
            'risk_score': risk_score,
            'updated_at': datetime.now()
        }
        self.users[user_id]['profile'] = profile
    
    def save_to_file(self, filepath='data/processed/database.pkl'):
        """Save database to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'users': self.users,
                'transactions': self.transactions,
                'wallets': self.wallets,
                'portfolios': self.portfolios,
                'investments': self.investments,
                'user_profiles': self.user_profiles
            }, f)
    
    def load_from_file(self, filepath='data/processed/database.pkl'):
        """Load database from file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.users = data['users']
                self.transactions = data['transactions']
                self.wallets = data['wallets']
                self.portfolios = data['portfolios']
                self.investments = data['investments']
                self.user_profiles = data['user_profiles']
            return True
        return False