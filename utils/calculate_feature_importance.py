import numpy as np
import pandas as pd

def calculate_feature_importance(pipeline):
    """Calculate and store feature importance"""
    try:
        # Get feature names from the preprocessor
        if hasattr(pipeline.named_steps['preprocessor'], 'transformers_'):
            num_features = []
            cat_features = []
            
            for name, trans, features in pipeline.named_steps['preprocessor'].transformers_:
                if name == 'num':
                    num_features = features
                elif name == 'cat':
                    if hasattr(trans, 'get_feature_names_out'):
                        cat_features = trans.get_feature_names_out(features)
                    else:
                        cat_features = features
            
            all_features = np.concatenate([num_features, cat_features])
        else:
            all_features = pipeline.named_steps['preprocessor'].get_feature_names_out()
        
        # Get importance scores
        if hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
            importances = pipeline.named_steps['regressor'].feature_importances_
        elif hasattr(pipeline.named_steps['regressor'], 'coef_'):
            importances = np.abs(pipeline.named_steps['regressor'].coef_)
        else:
            importances = np.ones(len(all_features)) / len(all_features)
        
        feature_importance = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
    except Exception as e:
        # self.logger.warning(f"Could not calculate feature importance: {str(e)}")
        feature_importance = None

        
    return feature_importance