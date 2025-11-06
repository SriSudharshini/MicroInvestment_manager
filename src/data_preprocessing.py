import pandas as pd
import numpy as np
from datetime import datetime

def load_and_preprocess_data(filepath, num_users=10):
    """
    Load transaction data and preprocess for the system
    
    Args:
        filepath: Path to CSV file
        num_users: Number of sample users to extract
    
    Returns:
        DataFrame with preprocessed transactions
    """
    # Load data
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False,nrows=50)
    
    # Select necessary columns
    columns_needed = [
        'trans_date_trans_time', 'cc_num', 'merchant', 
        'category', 'amt', 'first', 'last', 'gender', 
        'city', 'state', 'job', 'dob'
    ]
    
    # Check if all columns exist
    missing_cols = [col for col in columns_needed if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
        # Use only available columns
        columns_needed = [col for col in columns_needed if col in df.columns]
    
    df = df[columns_needed].copy()
    
    # Rename columns for clarity
    df.rename(columns={
        'trans_date_trans_time': 'timestamp',
        'cc_num': 'user_id',
        'amt': 'amount'
    }, inplace=True)
    
    # Convert timestamp - handle different date formats
    print("Converting timestamps...")
    try:
        # Try day-first format (DD-MM-YYYY HH:MM)
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
    except Exception as e:
        print(f"Date conversion attempt 1 failed: {e}")
        try:
            # Try with format specification
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M', errors='coerce')
        except:
            print("Trying mixed format...")
            # Last resort - let pandas infer
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Remove rows with invalid timestamps
    initial_count = len(df)
    df = df.dropna(subset=['timestamp'])
    print(f"Removed {initial_count - len(df)} rows with invalid timestamps")
    
    # Select top N users by transaction count
    print(f"Selecting top {num_users} users...")
    user_counts = df['user_id'].value_counts()
    top_users = user_counts.head(num_users).index.tolist()
    df = df[df['user_id'].isin(top_users)].copy()
    
    # Convert user_id to string for consistency
    df['user_id'] = df['user_id'].astype(str)
    
    # Sort by timestamp
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Create user name column
    if 'first' in df.columns and 'last' in df.columns:
        df['name'] = df['first'].astype(str) + ' ' + df['last'].astype(str)
    else:
        df['name'] = 'User ' + df['user_id'].astype(str)
    
    # Clean amount (ensure positive and numeric)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['amount'])
    df['amount'] = df['amount'].abs()
    
    # Remove zero amounts
    df = df[df['amount'] > 0].copy()
    
    print(f"Loaded {len(df)} transactions for {df['user_id'].nunique()} users")
    
    return df

def extract_user_info(df):
    """Extract unique user information"""
    agg_dict = {'name': 'first', 'user_id': 'first'}
    
    # Add optional fields if they exist
    for col in ['gender', 'city', 'state', 'job', 'dob']:
        if col in df.columns:
            agg_dict[col] = 'first'
    
    user_info = df.groupby('user_id').agg(agg_dict).reset_index(drop=True)
    
    return user_info

def compute_spending_features(transactions_df, user_id):
    """
    Compute spending features for user profiling
    
    Args:
        transactions_df: DataFrame of transactions
        user_id: User ID
    
    Returns:
        Dictionary of features
    """
    user_trans = transactions_df[transactions_df['user_id'] == user_id].copy()
    
    if len(user_trans) == 0:
        return None
    
    # Time-based aggregations
    user_trans['month'] = user_trans['timestamp'].dt.to_period('M')
    monthly_spend = user_trans.groupby('month')['amount'].sum()
    
    # Category-based spending (if category exists)
    if 'category' in user_trans.columns:
        category_spend = user_trans.groupby('category')['amount'].sum()
        total_spend = category_spend.sum()
        category_pct = (category_spend / total_spend * 100).to_dict() if total_spend > 0 else {}
    else:
        category_pct = {}
        total_spend = user_trans['amount'].sum()
    
    # Compute features
    features = {
        'avg_monthly_spend': monthly_spend.mean() if len(monthly_spend) > 0 else 0,
        'std_monthly_spend': monthly_spend.std() if len(monthly_spend) > 1 else 0,
        'avg_transaction_amount': user_trans['amount'].mean(),
        'std_transaction_amount': user_trans['amount'].std(),
        'transaction_count': len(user_trans),
        'unique_merchants': user_trans['merchant'].nunique() if 'merchant' in user_trans.columns else 0,
        'unique_categories': user_trans['category'].nunique() if 'category' in user_trans.columns else 0,
        'category_percentages': category_pct,
        'total_spend': total_spend,
        'max_transaction': user_trans['amount'].max(),
        'min_transaction': user_trans['amount'].min(),
    }
    
    # Regularity metric (transactions per day)
    date_range = (user_trans['timestamp'].max() - user_trans['timestamp'].min()).days
    features['transactions_per_day'] = len(user_trans) / max(date_range, 1)
    
    return features

def prepare_ml_features(features_dict):
    """
    Convert features dictionary to ML-ready format
    
    Args:
        features_dict: Dictionary of user features
    
    Returns:
        Numpy array of features
    """
    if features_dict is None:
        return None
    
    # Select numeric features for ML
    feature_vector = [
        features_dict['avg_monthly_spend'],
        features_dict['std_monthly_spend'],
        features_dict['avg_transaction_amount'],
        features_dict['std_transaction_amount'],
        features_dict['transaction_count'],
        features_dict['unique_merchants'],
        features_dict['unique_categories'],
        features_dict['transactions_per_day'],
        features_dict['max_transaction'],
    ]
    
    # Handle NaN values
    feature_vector = [0 if np.isnan(x) or np.isinf(x) else x for x in feature_vector]
    
    return np.array(feature_vector)

def create_sample_users_dataset(df, db):
    """
    Create sample users in the database from transaction data
    
    Args:
        df: Preprocessed transaction DataFrame
        db: Database instance
    
    Returns:
        List of user IDs
    """
    user_info = extract_user_info(df)
    user_ids = []
    
    for _, user in user_info.iterrows():
        user_id = str(user['user_id'])
        name = user.get('name', f'User {user_id[:8]}')
        db.add_user(user_id, name)
        user_ids.append(user_id)
    
    print(f"Created {len(user_ids)} users in database")
    
    return user_ids