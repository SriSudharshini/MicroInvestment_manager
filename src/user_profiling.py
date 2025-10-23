import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import os

class UserProfiler:
    """
    ML-based user profiling system using clustering
    """
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.pca = PCA(n_components=2, random_state=42)
        self.is_fitted = False
        self.cluster_labels = ['Conservative', 'Moderate', 'Aggressive']
        self.cluster_mapping = {}
        
    def fit(self, feature_matrix, user_ids):
        """
        Fit the profiling model on user features
        
        Args:
            feature_matrix: numpy array of shape (n_users, n_features)
            user_ids: list of user IDs
        
        Returns:
            Dictionary mapping user_id to profile
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Fit clustering
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Fit PCA for visualization
        self.pca.fit(X_scaled)
        
        # Map clusters to risk profiles
        # Higher spending + higher variance = Aggressive
        # Lower spending + lower variance = Conservative
        cluster_stats = []
        for i in range(self.n_clusters):
            cluster_mask = clusters == i
            cluster_features = feature_matrix[cluster_mask]
            
            avg_spend = cluster_features[:, 0].mean()  # avg_monthly_spend
            std_spend = cluster_features[:, 1].mean()  # std_monthly_spend
            avg_trans = cluster_features[:, 2].mean()  # avg_transaction_amount
            
            risk_score = (avg_spend + std_spend + avg_trans) / 3
            cluster_stats.append((i, risk_score))
        
        # Sort by risk score and assign labels
        cluster_stats.sort(key=lambda x: x[1])
        for idx, (cluster_id, _) in enumerate(cluster_stats):
            self.cluster_mapping[cluster_id] = self.cluster_labels[idx]
        
        # Create user profiles
        user_profiles = {}
        for user_id, cluster in zip(user_ids, clusters):
            profile = self.cluster_mapping[cluster]
            risk_score = self._compute_risk_score(cluster, cluster_stats)
            user_profiles[user_id] = {
                'profile': profile,
                'cluster': int(cluster),
                'risk_score': risk_score
            }
        
        self.is_fitted = True
        return user_profiles
    
    def _compute_risk_score(self, cluster, cluster_stats):
        """Compute normalized risk score (0-1) for a cluster"""
        scores = [score for _, score in cluster_stats]
        min_score, max_score = min(scores), max(scores)
        
        cluster_score = next(score for cid, score in cluster_stats if cid == cluster)
        
        if max_score == min_score:
            return 0.5
        
        normalized = (cluster_score - min_score) / (max_score - min_score)
        return normalized
    
    def predict(self, feature_vector):
        """
        Predict profile for a new user
        
        Args:
            feature_vector: numpy array of features
        
        Returns:
            Dictionary with profile and risk_score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        cluster = self.kmeans.predict(X_scaled)[0]
        
        profile = self.cluster_mapping[cluster]
        
        # Compute risk score based on distance to cluster centers
        distances = self.kmeans.transform(X_scaled)[0]
        risk_score = 1 - (distances[cluster] / distances.sum())
        
        return {
            'profile': profile,
            'cluster': int(cluster),
            'risk_score': risk_score
        }
    
    def get_pca_projection(self, feature_matrix):
        """Get 2D PCA projection for visualization"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.scaler.transform(feature_matrix)
        return self.pca.transform(X_scaled)
    
    def save_model(self, filepath='models/saved_models/user_profiler.pkl'):
        """Save the model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'pca': self.pca,
            'cluster_mapping': self.cluster_mapping,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath='models/saved_models/user_profiler.pkl'):
        """Load the model from disk"""
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.kmeans = model_data['kmeans']
        self.pca = model_data['pca']
        self.cluster_mapping = model_data['cluster_mapping']
        self.is_fitted = model_data['is_fitted']
        
        return True

def build_user_profiles(transactions_df, user_ids, db):
    """
    Build user profiles for all users and store in database
    
    Args:
        transactions_df: DataFrame of transactions
        user_ids: List of user IDs
        db: Database instance
    
    Returns:
        UserProfiler instance
    """
    from src.data_preprocessing import compute_spending_features, prepare_ml_features
    
    # Compute features for all users
    features_list = []
    valid_user_ids = []
    
    for user_id in user_ids:
        features_dict = compute_spending_features(transactions_df, user_id)
        if features_dict is not None:
            feature_vector = prepare_ml_features(features_dict)
            if feature_vector is not None:
                features_list.append(feature_vector)
                valid_user_ids.append(user_id)
    
    # Convert to matrix
    feature_matrix = np.array(features_list)
    
    # Fit profiler
    profiler = UserProfiler(n_clusters=3)
    user_profiles = profiler.fit(feature_matrix, valid_user_ids)
    
    # Store profiles in database
    for user_id, profile_data in user_profiles.items():
        db.set_user_profile(
            user_id, 
            profile_data['profile'], 
            profile_data['risk_score']
        )
    
    return profiler