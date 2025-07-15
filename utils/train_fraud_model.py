from sklearn.ensemble import IsolationForest
import numpy as np

def train_fraud_model(X, y, preprocessor):
    """Train isolation forest for fraud detection"""
    try:
        # Use preprocessor to transform data
        X_transformed = preprocessor.transform(X)
        
        # Train isolation forest
        fraud_model = IsolationForest(
            n_estimators=100,
            contamination=0.01,
            random_state=42
        )
        fraud_model.fit(X_transformed)
        
        # Set dynamic threshold based on claim amounts
        scores = fraud_model.decision_function(X_transformed)
        fraud_threshold = np.percentile(scores, 1)
        
    except Exception as e:
        print(str(e))