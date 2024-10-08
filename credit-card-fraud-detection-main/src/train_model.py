from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def train_logistic_regression(X_train, y_train):
    # Logistic Regression Model
    lr = LogisticRegression(random_state=42)
    
    # Using SMOTE for handling imbalanced classes
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    lr.fit(X_res, y_res)
    return lr

def train_decision_tree(X_train, y_train):
    # Decision Tree Model
    dt = DecisionTreeClassifier(random_state=42)
    
    # Using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    dt.fit(X_res, y_res)
    return dt

def train_random_forest(X_train, y_train):
    # Random Forest Model
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    
    # Using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    rf.fit(X_res, y_res)
    return rf
