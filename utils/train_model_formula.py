from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np
import optuna
from utils.preprocess_data import preprocess_data
from utils.calculate_feature_importance import calculate_feature_importance
from utils.train_fraud_model import train_fraud_model
from datetime import datetime
import traceback
from utils.model_monitor import ModelMonitor

def train_model_formula(model_algorithm, target_variable, test_set_size, cross_validation_folds, enable_hyperparameter_tuning, max_iter, records):
    """Enhanced model training with multiple algorithms"""
    try:
        X, y, preprocessor, required_prediction_columns = preprocess_data(target_variable, records)

        if X is None or y is None:
            return False
        
        # Split data with temporal validation if date is available
        if 'Submission_Date' in X.columns:
            X_sorted = X.sort_values('Submission_Date')
            y_sorted = y[X_sorted.index]
            split_idx = int(len(X_sorted) * (1 - test_set_size))
            X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
            y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_set_size=test_set_size, random_state=42)
    
        # Initialize models
        models = {}
        param_grids = {}
    
        if model_algorithm == "Gradient Boosting" or model_algorithm == "Auto Select Best":
            models['GradientBoosting'] = GradientBoostingRegressor(random_state=42)
            param_grids['GradientBoosting'] = {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__learning_rate': [0.01, 0.05, 0.1],
                'regressor__max_depth': [3, 5, 7]
            }
    
        if model_algorithm == "Random Forest" or model_algorithm == "Auto Select Best":
            models['RandomForest'] = RandomForestRegressor(random_state=42)
            param_grids['RandomForest'] = {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [5, 10, None],
                'regressor__min_samples_split': [2, 5, 10]
            }
    
        if model_algorithm == "XGBoost" or model_algorithm == "Auto Select Best":
            models['XGBoost'] = xgb.XGBRegressor(random_state=42)
            param_grids['XGBoost'] = {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [3, 5, 7],
                'regressor__learning_rate': [0.01, 0.05, 0.1],
                'regressor__tree_method': ['gpu_hist']  # Enable GPU acceleration
            }
    
        if model_algorithm == "Neural Network" or model_algorithm == "Auto Select Best":
            models['NeuralNetwork'] = MLPRegressor(random_state=42, max_iter=500)
            param_grids['NeuralNetwork'] = {
                'regressor__hidden_layer_sizes': [(50,), (50, 25), (100, 50)],
                'regressor__activation': ['relu', 'tanh'],
                'regressor__solver': ['adam', 'sgd']
            }

        # Train and evaluate each model
        results = []
        best_model = None
        best_score = -np.inf
    
        for name, model in models.items():
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
        
            # Hyperparameter tuning if enabled
            if enable_hyperparameter_tuning and name in param_grids:
                if name == 'XGBoost':
                    # Special handling for XGBoost with Optuna
                    def objective(trial):
                        params = {
                            'regressor__n_estimators': trial.suggest_int('regressor__n_estimators', 50, 500),
                            'regressor__max_depth': trial.suggest_int('regressor__max_depth', 3, 10),
                            'regressor__learning_rate': trial.suggest_float('regressor__learning_rate', 0.001, 0.1, log=True),
                            'regressor__subsample': trial.suggest_float('regressor__subsample', 0.5, 1.0),
                            'regressor__colsample_bytree': trial.suggest_float('regressor__colsample_bytree', 0.5, 1.0)
                        }
                        pipeline.set_params(**params)
                        pipeline.fit(X_train, y_train)
                        return mean_absolute_error(y_test, pipeline.predict(X_test))
                
                    study = optuna.create_study(direction='minimize')
                    study.optimize(objective, n_trials=max_iter)
                    best_params = study.best_params
                    pipeline.set_params(**best_params)
                else:
                    # Standard RandomizedSearchCV for other models
                    search = RandomizedSearchCV(
                        pipeline, 
                        param_grids[name], 
                        n_iter=max_iter, 
                        cv=cross_validation_folds,
                        scoring='neg_mean_absolute_error', 
                        random_state=42
                    )
                    
                    try:
                        # attempt to fit your search (GridSearchCV, RandomizedSearchCV, etc.)
                        search.fit(X_train, y_train)
                    except Exception as e:
                        # print full traceback for more context
                        traceback.print_exc()

                    pipeline = search.best_estimator_
                    # self.logger.info(f"Best params for {name}: {search.best_params_}")
        
            # Train model
            pipeline.fit(X_train, y_train)
            
            missing_cols = set(required_prediction_columns) - set(X_test.columns)
            if missing_cols:
                raise ValueError(f"❗ X_test is missing required columns: {missing_cols}")
                
            # Evaluate
            try:
                y_pred = pipeline.predict(X_test)
            except Exception as e:
                traceback.print_exc()
                
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results.append({
                'Model': name,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            })
        
            # Track best model
            if r2 > best_score:
                best_score = r2
                best_model = pipeline

        # Create results DataFrame
        results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
    
        if model_algorithm == "Auto Select Best":
            model = best_model
            baseline_metrics = {
                'MAE': results_df.iloc[0]['MAE'],
                'RMSE': results_df.iloc[0]['RMSE'],
                'R2': results_df.iloc[0]['R2']
            }
        else:
            # For single model selection, use the last trained pipeline
            model = pipeline
            baseline_metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
        
        # Calculate feature importance
        calculate_feature_importance(model)
        
        # Train fraud detection model
        train_fraud_model(X, y, preprocessor)

        # Log model performance
        monitor = ModelMonitor()
        monitor.log_performance(
            model_algorithm,
            {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        )

        # Set training date
        training_date = datetime.now()
    
        return results_df
    
    except Exception as e:
        # self.logger.error(f"Training failed: {str(e)}")
        # st.error(f"Training failed: {str(e)}")
        raise str(e)
        