import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pickle
import os

class UserProfiler:
    """
    ML-based user profiling system using multiple clustering algorithms
    Includes explainability features for transparency
    """
    
    def __init__(self, n_clusters=3, model_type='kmeans'):
        self.n_clusters = n_clusters
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        # Initialize both models for comparison
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
        
        self.pca = PCA(n_components=2, random_state=42)
        self.is_fitted = False
        self.cluster_labels = ['Conservative', 'Moderate', 'Aggressive']
        self.cluster_mapping = {}
        self.feature_names = [
            'avg_monthly_spend',
            'std_monthly_spend',
            'avg_transaction_amount',
            'std_transaction_amount',
            'transaction_count',
            'unique_merchants',
            'unique_categories',
            'transactions_per_day',
            'max_transaction'
        ]
        
        # Store both model results for comparison
        self.kmeans_profiles = {}
        self.gmm_profiles = {}
        self.comparison_metrics = {}
        
        # Feature importance (for explainability)
        self.feature_importance = {}
        
    def fit(self, feature_matrix, user_ids):
        """
        Fit both profiling models on user features and compare
        
        Args:
            feature_matrix: numpy array of shape (n_users, n_features)
            user_ids: list of user IDs
        
        Returns:
            Dictionary with both model results
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Fit both models
        kmeans_clusters = self.kmeans.fit_predict(X_scaled)
        gmm_clusters = self.gmm.fit_predict(X_scaled)
        
        # Calculate model comparison metrics
        self.comparison_metrics = {
            'kmeans': {
                'silhouette': silhouette_score(X_scaled, kmeans_clusters),
                'davies_bouldin': davies_bouldin_score(X_scaled, kmeans_clusters),
                'inertia': self.kmeans.inertia_
            },
            'gmm': {
                'silhouette': silhouette_score(X_scaled, gmm_clusters),
                'davies_bouldin': davies_bouldin_score(X_scaled, gmm_clusters),
                'bic': self.gmm.bic(X_scaled),
                'aic': self.gmm.aic(X_scaled)
            }
        }
        
        # Fit PCA for visualization
        self.pca.fit(X_scaled)
        
        # Map clusters to risk profiles for both models
        self._map_clusters_to_profiles(feature_matrix, kmeans_clusters, 'kmeans')
        self._map_clusters_to_profiles(feature_matrix, gmm_clusters, 'gmm')
        
        # Calculate feature importance for explainability
        self._calculate_feature_importance(X_scaled, kmeans_clusters)
        
        # Create user profiles for both models
        for user_id, km_cluster, gm_cluster in zip(user_ids, kmeans_clusters, gmm_clusters):
            # K-Means profile
            km_profile = self.cluster_mapping['kmeans'][km_cluster]
            km_risk_score = self._compute_risk_score(km_cluster, feature_matrix, kmeans_clusters)
            
            self.kmeans_profiles[user_id] = {
                'profile': km_profile,
                'cluster': int(km_cluster),
                'risk_score': km_risk_score,
                'model': 'K-Means'
            }
            
            # GMM profile
            gm_profile = self.cluster_mapping['gmm'][gm_cluster]
            gm_risk_score = self._compute_risk_score(gm_cluster, feature_matrix, gmm_clusters)
            
            # GMM also provides probability distribution
            probs = self.gmm.predict_proba(self.scaler.transform([feature_matrix[list(user_ids).index(user_id)]]))[0]
            
            self.gmm_profiles[user_id] = {
                'profile': gm_profile,
                'cluster': int(gm_cluster),
                'risk_score': gm_risk_score,
                'model': 'Gaussian Mixture',
                'cluster_probabilities': probs.tolist()
            }
        
        self.is_fitted = True
        
        # Return the primary model's profiles (user selectable)
        if self.model_type == 'kmeans':
            return self.kmeans_profiles
        else:
            return self.gmm_profiles
    
    def _map_clusters_to_profiles(self, feature_matrix, clusters, model_name):
        """Map clusters to risk profiles based on spending patterns"""
        cluster_stats = []
        for i in range(self.n_clusters):
            cluster_mask = clusters == i
            cluster_features = feature_matrix[cluster_mask]
            
            if len(cluster_features) > 0:
                avg_spend = cluster_features[:, 0].mean()
                std_spend = cluster_features[:, 1].mean()
                avg_trans = cluster_features[:, 2].mean()
                
                risk_score = (avg_spend + std_spend + avg_trans) / 3
                cluster_stats.append((i, risk_score))
            else:
                cluster_stats.append((i, 0))
        
        # Sort by risk score and assign labels
        cluster_stats.sort(key=lambda x: x[1])
        
        if model_name not in self.cluster_mapping:
            self.cluster_mapping[model_name] = {}
        
        for idx, (cluster_id, _) in enumerate(cluster_stats):
            self.cluster_mapping[model_name][cluster_id] = self.cluster_labels[idx]
    
    def _compute_risk_score(self, cluster, feature_matrix, clusters):
        """Compute normalized risk score (0-1) for a cluster"""
        cluster_means = []
        for i in range(self.n_clusters):
            cluster_mask = clusters == i
            if cluster_mask.any():
                cluster_mean = feature_matrix[cluster_mask].mean(axis=0)
                risk_metric = (cluster_mean[0] + cluster_mean[1] + cluster_mean[2]) / 3
                cluster_means.append(risk_metric)
        
        if len(cluster_means) == 0:
            return 0.5
        
        min_risk = min(cluster_means)
        max_risk = max(cluster_means)
        
        if max_risk == min_risk:
            return 0.5
        
        cluster_mask = clusters == cluster
        if not cluster_mask.any():
            return 0.5
            
        cluster_mean = feature_matrix[cluster_mask].mean(axis=0)
        cluster_risk = (cluster_mean[0] + cluster_mean[1] + cluster_mean[2]) / 3
        
        normalized = (cluster_risk - min_risk) / (max_risk - min_risk)
        return normalized
    
    def _calculate_feature_importance(self, X_scaled, clusters):
        """
        Calculate feature importance for explainability
        Based on how much each feature contributes to cluster separation
        """
        importance_scores = []
        
        for i in range(X_scaled.shape[1]):
            feature_values = X_scaled[:, i]
            
            # Calculate between-cluster variance / within-cluster variance
            cluster_means = []
            cluster_vars = []
            
            for c in range(self.n_clusters):
                cluster_mask = clusters == c
                if cluster_mask.any():
                    cluster_data = feature_values[cluster_mask]
                    cluster_means.append(cluster_data.mean())
                    cluster_vars.append(cluster_data.var())
            
            if len(cluster_means) > 0 and len(cluster_vars) > 0:
                between_var = np.var(cluster_means)
                within_var = np.mean(cluster_vars)
                
                # F-ratio like score
                if within_var > 0:
                    importance = between_var / within_var
                else:
                    importance = 0
            else:
                importance = 0
            
            importance_scores.append(importance)
        
        # Normalize to sum to 1
        total_importance = sum(importance_scores)
        if total_importance > 0:
            importance_scores = [score / total_importance for score in importance_scores]
        
        self.feature_importance = {
            name: score 
            for name, score in zip(self.feature_names, importance_scores)
        }
    
    def get_feature_importance(self):
        """Get feature importance for explainability"""
        return self.feature_importance
    
    def explain_user_profile(self, user_id, feature_vector):
        """
        Explain why a user got their specific profile
        XAI component - shows which features contributed most
        """
        if not self.is_fitted:
            return "Model not fitted yet"
        
        # Get user's profile
        if self.model_type == 'kmeans':
            profile_data = self.kmeans_profiles.get(user_id)
        else:
            profile_data = self.gmm_profiles.get(user_id)
        
        if not profile_data:
            return "User profile not found"
        
        # Normalize feature vector
        feature_vector_norm = (feature_vector - feature_vector.mean()) / (feature_vector.std() + 1e-10)
        
        # Calculate contribution of each feature
        contributions = []
        for i, (name, value) in enumerate(zip(self.feature_names, feature_vector)):
            importance = self.feature_importance.get(name, 0)
            contribution = abs(feature_vector_norm[i]) * importance
            contributions.append({
                'feature': name,
                'value': value,
                'importance': importance,
                'contribution': contribution
            })
        
        # Sort by contribution
        contributions.sort(key=lambda x: x['contribution'], reverse=True)
        
        # Create explanation
        explanation = f"**{profile_data['profile']} Profile Explanation**\n\n"
        explanation += f"Risk Score: {profile_data['risk_score']:.2f}\n\n"
        explanation += "**Top Contributing Factors:**\n\n"
        
        for i, contrib in enumerate(contributions[:5], 1):
            feature_display = contrib['feature'].replace('_', ' ').title()
            explanation += f"{i}. **{feature_display}**: {contrib['value']:.2f}\n"
            explanation += f"   - Importance: {contrib['importance']:.1%}\n"
            explanation += f"   - Contribution: {contrib['contribution']:.3f}\n\n"
        
        return explanation
    
    def get_model_comparison(self):
        """Get comparison between K-Means and GMM models"""
        return self.comparison_metrics
    
    def get_both_profiles(self, user_id):
        """Get profiles from both models for comparison"""
        return {
            'kmeans': self.kmeans_profiles.get(user_id, {}),
            'gmm': self.gmm_profiles.get(user_id, {})
        }
    
    def predict(self, feature_vector):
        """Predict profile for a new user"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        
        if self.model_type == 'kmeans':
            cluster = self.kmeans.predict(X_scaled)[0]
            profile = self.cluster_mapping['kmeans'][cluster]
            distances = self.kmeans.transform(X_scaled)[0]
            risk_score = 1 - (distances[cluster] / distances.sum())
        else:
            cluster = self.gmm.predict(X_scaled)[0]
            profile = self.cluster_mapping['gmm'][cluster]
            probs = self.gmm.predict_proba(X_scaled)[0]
            risk_score = probs[cluster]
        
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
            'gmm': self.gmm,
            'pca': self.pca,
            'cluster_mapping': self.cluster_mapping,
            'is_fitted': self.is_fitted,
            'kmeans_profiles': self.kmeans_profiles,
            'gmm_profiles': self.gmm_profiles,
            'comparison_metrics': self.comparison_metrics,
            'feature_importance': self.feature_importance
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
        self.gmm = model_data['gmm']
        self.pca = model_data['pca']
        self.cluster_mapping = model_data['cluster_mapping']
        self.is_fitted = model_data['is_fitted']
        self.kmeans_profiles = model_data.get('kmeans_profiles', {})
        self.gmm_profiles = model_data.get('gmm_profiles', {})
        self.comparison_metrics = model_data.get('comparison_metrics', {})
        self.feature_importance = model_data.get('feature_importance', {})
        
        return True

def build_user_profiles(transactions_df, user_ids, db, model_type='kmeans'):
    """
    Build user profiles for all users and store in database
    
    Args:
        transactions_df: DataFrame of transactions
        user_ids: List of user IDs
        db: Database instance
        model_type: 'kmeans' or 'gmm'
    
    Returns:
        UserProfiler instance
    """
    from src.data_preprocessing import compute_spending_features, prepare_ml_features
    
    # Compute features for all users
    features_list = []
    valid_user_ids = []
    features_dict_list = []
    
    for user_id in user_ids:
        features_dict = compute_spending_features(transactions_df, user_id)
        if features_dict is not None:
            feature_vector = prepare_ml_features(features_dict)
            if feature_vector is not None:
                features_list.append(feature_vector)
                valid_user_ids.append(user_id)
                features_dict_list.append(features_dict)
    
    # Convert to matrix
    feature_matrix = np.array(features_list)
    
    # Fit profiler with both models
    profiler = UserProfiler(n_clusters=3, model_type=model_type)
    user_profiles = profiler.fit(feature_matrix, valid_user_ids)
    
    # Store profiles in database (primary model)
    for user_id, profile_data in user_profiles.items():
        db.set_user_profile(
            user_id, 
            profile_data['profile'], 
            profile_data['risk_score']
        )
    
    # Store feature vectors for explainability
    db.user_feature_vectors = {
        uid: features_dict_list[i] 
        for i, uid in enumerate(valid_user_ids)
    }
    db.user_feature_matrices = {
        uid: features_list[i]
        for i, uid in enumerate(valid_user_ids)
    }
    
    return profiler