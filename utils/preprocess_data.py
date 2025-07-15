from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(target_variable, records):
    """Prepare data for modeling with enhanced validation"""
    try:    
        # Define features and target_variable
        X = records.drop(columns=[target_variable], errors='ignore')
        y = records[target_variable]

        # Define feature types with validation
        categorical_features = [
            'Visit_Type', 'Diagnosis_Group', 'Treatment_Type', 
            'Provider_Name', 'Hospital_County', 'Employee_Gender',
            'Claim_Weekday', 'Claim_Month', 'Employer', 'Category',
            'Age_Group', 'Claim_Size', 'Department', 'Tenure_Group'
        ]
        
        numerical_features = [
            'Employee_Age', 'Co_Payment_KES', 'Is_Pre_Authorized',
            'Inpatient_Cap_KES_Utilization', 'Outpatient_Cap_KES_Utilization',
            'Optical_Cap_KES_Utilization', 'Dental_Cap_KES_Utilization',
            'Maternity_Cap_KES_Utilization', 'Claim_Amount_to_Mean',
            'Same_Day_Claims', 'Employer_Z_Score'
        ]
        
        # Validate features exist in data
        categorical_features = [f for f in categorical_features if f in X.columns]
        numerical_features = [f for f in numerical_features if f in X.columns]
        
        if not categorical_features and not numerical_features:
            raise ValueError("No valid features found for modeling")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
        # Store required columns for prediction
        required_prediction_columns = numerical_features + categorical_features
        
        return X, y, preprocessor, required_prediction_columns
        
    except Exception as e:
        return None, None, None, None