import joblib

def save_model(model, filename):
    # Save the model to disk
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")
