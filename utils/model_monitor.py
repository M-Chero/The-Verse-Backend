import numpy as np
from datetime import datetime
from scipy.stats import ks_2samp

class ModelMonitor:
    """Class for monitoring model performance and data drift"""
    def __init__(self):
        self.performance_history = []
        self.data_drift_scores = []
        self.training_date = datetime.now()
        
    def log_performance(self, model_name, metrics):
        """Log model performance metrics"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'model': model_name,
            **metrics
        })
        
    def check_data_drift(self, current_data, reference_data):
        """Calculate data drift metrics"""
        drift_metrics = {}
        
        # For numerical columns
        num_cols = current_data.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            # Kolmogorov-Smirnov test
            stat, p = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
            drift_metrics[col] = {'ks_stat': stat, 'ks_p': p}
        
        # For categorical columns
        cat_cols = current_data.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            # Population Stability Index
            ref_counts = reference_data[col].value_counts(normalize=True)
            curr_counts = current_data[col].value_counts(normalize=True)
            psi = np.sum((curr_counts - ref_counts) * np.log(curr_counts / ref_counts))
            drift_metrics[col] = {'psi': psi}
        
        self.data_drift_scores.append({
            'timestamp': datetime.now(),
            'drift_metrics': drift_metrics
        })

        return drift_metrics