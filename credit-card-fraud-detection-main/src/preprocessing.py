import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path):
    # Load the training and test data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, test_data

def preprocess_data(df):
    # Drop non-numeric columns or convert them to numeric
    df = df.select_dtypes(include=['number'])  # Keep only numeric columns
    
    # Separate features and target variable
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud']
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    return X_train, X_val, y_train, y_val

