from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_val, y_val):
    # Predict on validation data
    y_pred = model.predict(X_val)
    
    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_val, y_pred)}")
    print(f"Confusion Matrix: \n{confusion_matrix(y_val, y_pred)}")
    print(f"Classification Report: \n{classification_report(y_val, y_pred)}")
    print(f"ROC-AUC: {roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])}")
