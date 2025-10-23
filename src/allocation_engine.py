import numpy as np
from datetime import datetime, timedelta

class AllocationEngine:
    """
    Allocation engine with rule-based baseline and ML-driven adjustments
    """
    
    def __init__(self):
        # Rule-based baseline allocations
        self.baseline_allocations = {
            'Conservative': {
                'equity': 0.10,
                'gold': 0.30,
                'fd': 0.50,
                'liquid': 0.10
            },
            'Moderate': {
                'equity': 0.40,
                'gold': 0.25,
                'fd': 0.25,
                'liquid': 0.10
            },
            'Aggressive': {
                'equity': 0.65,
                'gold': 0.20,
                'fd': 0.10,
                'liquid': 0.05
            }
        }
        
        # ML adjustment parameters
        self.learning_rate = 0.03  # 3% adjustment per update
        self.max_shift = 0.10  # Max 10% change per update
        self.min_liquid = 0.05  # Minimum 5% in liquid
        self.max_equity = 0.80  # Maximum 80% in equity
        
        # Store custom allocations per user
        self.custom_allocations = {}
        
        # Performance history for learning
        self.performance_history = {}
    
    def get_allocation(self, user_id, profile, wallet_amount):
        """
        Get allocation for a user based on profile
        
        Args:
            user_id: User ID
            profile: Risk profile (Conservative/Moderate/Aggressive)
            wallet_amount: Amount to allocate
        
        Returns:
            Dictionary with asset: amount pairs
        """
        # Check if user has custom learned allocation
        if user_id in self.custom_allocations:
            weights = self.custom_allocations[user_id]
        else:
            weights = self.baseline_allocations[profile].copy()
        
        # Calculate amounts
        allocation = {}
        for asset, weight in weights.items():
            allocation[asset] = round(wallet_amount * weight, 2)
        
        # Ensure total equals wallet_amount (handle rounding)
        total = sum(allocation.values())
        if total != wallet_amount:
            diff = wallet_amount - total
            allocation['liquid'] += diff
        
        return allocation
    
    def update_weights(self, user_id, profile, performance_data):
        """
        Update allocation weights based on performance using simple learning
        
        Args:
            user_id: User ID
            profile: Current risk profile
            performance_data: Dict with asset: return_percentage
        
        Returns:
            Updated weights dictionary
        """
        # Get current weights
        if user_id in self.custom_allocations:
            current_weights = self.custom_allocations[user_id].copy()
        else:
            current_weights = self.baseline_allocations[profile].copy()
        
        # Store performance history
        if user_id not in self.performance_history:
            self.performance_history[user_id] = []
        
        self.performance_history[user_id].append({
            'timestamp': datetime.now(),
            'performance': performance_data.copy()
        })
        
        # Calculate average performance across assets
        avg_performance = np.mean(list(performance_data.values()))
        
        # Adjust weights based on relative performance
        new_weights = {}
        adjustments = {}
        
        for asset, current_weight in current_weights.items():
            if asset in performance_data:
                # Performance delta (how much better/worse than average)
                perf_delta = performance_data[asset] - avg_performance
                
                # Calculate adjustment (bounded by learning rate)
                adjustment = self.learning_rate * perf_delta / 100
                adjustment = np.clip(adjustment, -self.max_shift, self.max_shift)
                
                # Apply adjustment
                new_weight = current_weight * (1 + adjustment)
                adjustments[asset] = adjustment
                new_weights[asset] = new_weight
            else:
                new_weights[asset] = current_weight
        
        # Normalize weights to sum to 1
        total_weight = sum(new_weights.values())
        for asset in new_weights:
            new_weights[asset] /= total_weight
        
        # Apply constraints
        new_weights = self._apply_constraints(new_weights)
        
        # Store updated weights
        self.custom_allocations[user_id] = new_weights
        
        return new_weights, adjustments
    
    def _apply_constraints(self, weights):
        """Apply safety constraints on weights"""
        # Ensure minimum liquid allocation
        if weights['liquid'] < self.min_liquid:
            deficit = self.min_liquid - weights['liquid']
            weights['liquid'] = self.min_liquid
            
            # Reduce other assets proportionally
            other_assets = [a for a in weights if a != 'liquid']
            total_other = sum(weights[a] for a in other_assets)
            
            if total_other > 0:
                for asset in other_assets:
                    weights[asset] -= (weights[asset] / total_other) * deficit
        
        # Cap equity at maximum
        if weights['equity'] > self.max_equity:
            excess = weights['equity'] - self.max_equity
            weights['equity'] = self.max_equity
            
            # Distribute excess to other assets
            other_assets = [a for a in weights if a != 'equity']
            for asset in other_assets:
                weights[asset] += excess / len(other_assets)
        
        # Ensure all weights are non-negative
        for asset in weights:
            weights[asset] = max(0, weights[asset])
        
        # Renormalize
        total = sum(weights.values())
        if total > 0:
            for asset in weights:
                weights[asset] /= total
        
        return weights
    
    def get_allocation_explanation(self, user_id, profile):
        """Generate human-readable explanation of allocation"""
        if user_id in self.custom_allocations:
            weights = self.custom_allocations[user_id]
            explanation = f"ML-adjusted allocation based on your {profile} profile and past performance:\n"
            
            if user_id in self.performance_history and len(self.performance_history[user_id]) > 0:
                last_perf = self.performance_history[user_id][-1]['performance']
                best_asset = max(last_perf.items(), key=lambda x: x[1])[0]
                explanation += f"Recently, {best_asset} performed best, so we slightly increased its weight.\n"
        else:
            weights = self.baseline_allocations[profile]
            explanation = f"Standard {profile} allocation strategy:\n"
        
        for asset, weight in weights.items():
            explanation += f"  • {asset.capitalize()}: {weight*100:.1f}%\n"
        
        return explanation

def check_batch_trigger(db, user_id):
    """
    Check if batch investment should be triggered
    
    Args:
        db: Database instance
        user_id: User ID
    
    Returns:
        Boolean indicating if investment should occur
    """
    wallet_balance = db.get_wallet_balance(user_id)
    threshold = db.users[user_id]['threshold']
    
    return wallet_balance >= threshold