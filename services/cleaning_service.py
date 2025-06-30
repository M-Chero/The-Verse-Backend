import pandas as pd
import numpy as np
from datetime import datetime

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # ====== Data Type Validation ======
    type_conversions = {
        'Employee_Age': 'int',
        'Claim_Amount_KES': 'float',
        'Co_Payment_KES': 'float',
        'Submission_Date': 'datetime64[ns]',
        'Service_Date': 'datetime64[ns]',
        'Hire_Date': 'datetime64[ns]'
    }
    for col, dtype in type_conversions.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                if dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                else:
                    df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'(\\d+\\.?\\d*)')[0], errors='coerce')

    # ====== Group-Specific Features ======
    if 'Employer' in df.columns:
        df['Employer'] = df['Employer'].str.upper().str.strip()

    if 'Department' not in df.columns and 'Division' in df.columns:
        df['Department'] = df['Division'].str.title()
    else:
        df['Department'] = 'General'

    if 'Hire_Date' in df.columns:
        df['Tenure'] = (datetime.now() - df['Hire_Date']).dt.days / 365
        df['Tenure_Group'] = pd.cut(df['Tenure'], bins=[0, 1, 5, 100], labels=['<1yr', '1-5yrs', '5+yrs'])

    if 'Salary' in df.columns:
        df['Salary_Band'] = pd.qcut(df['Salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # ====== Missing Values ======
    missing_report = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
    missing_report['% Missing'] = (missing_report['Missing Values'] / len(df)) * 100
    cols_to_drop = missing_report[missing_report['% Missing'] > 70].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        elif df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())

    # ====== Outlier Handling ======
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if col in ['Employee_ID']: continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        outliers = (df[col] < lower) | (df[col] > upper) | (z_scores.abs() > 3)
        if outliers.any():
            df[f'{col}_outlier'] = outliers.astype(int)
            df[col] = np.where(outliers, df[col].median(), df[col])

    # ====== Value Corrections ======
    if 'Claim_Amount_KES' in df.columns:
        df['Claim_Amount_KES'] = df['Claim_Amount_KES'].abs()
    if 'Employee_Age' in df.columns:
        df['Employee_Age'] = df['Employee_Age'].apply(lambda x: x if 18 <= x <= 100 else np.nan).fillna(df['Employee_Age'].median())

    # ====== Deduplication ======
    dup_cols = [c for c in df.columns if c not in ['Claim_Amount_KES', 'Submission_Date']]
    df = df.drop_duplicates(subset=dup_cols, keep='first')

    # ====== Categorical Normalization ======
    categorical_cols = ['Visit_Type', 'Provider_Name', 'Hospital_County', 'Employee_Gender', 'Category', 'Employer', 'Department']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.title().str.strip()
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].apply(lambda x: x if freq.get(x, 0) > 0.05 else 'Other')

    # ====== Format Corrections ======
    for col in df.columns:
        if '_KES' in col:
            df[col] = df[col].replace('[^\\d.]', '', regex=True).astype(float)
        if 'Date' in col or 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # ====== Feature Engineering ======
    if 'Pre_Authorization_Required' in df.columns:
        df['Is_Pre_Authorized'] = df['Pre_Authorization_Required'].map({'Yes': 1, 'No': 0})

    for col in ['Inpatient_Cap_KES', 'Outpatient_Cap_KES', 'Optical_Cap_KES', 'Dental_Cap_KES', 'Maternity_Cap_KES']:
        if col in df.columns:
            df[f'{col}_Utilization'] = df['Claim_Amount_KES'] / df[col].replace(0, np.nan)

    if 'Diagnosis' in df.columns:
        df['Diagnosis_Group'] = df['Diagnosis'].str.extract(r'([A-Za-z\\s]+)')[0].str.strip()
        df['Diagnosis_Group'] = df['Diagnosis_Group'].apply(lambda x: x if len(str(x)) > 3 else 'Other')

    if 'Treatment' in df.columns:
        df['Treatment_Type'] = df['Treatment'].str.extract(r'([A-Za-z\\s]+)')[0].str.strip()
        df['Treatment_Type'] = df['Treatment_Type'].apply(lambda x: x if len(str(x)) > 3 else 'Other')

    if 'Employee_Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Employee_Age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])

    if 'Claim_Amount_KES' in df.columns:
        df['Claim_Size'] = pd.qcut(df['Claim_Amount_KES'], q=4, labels=['Small', 'Medium', 'Large', 'Very Large'])

    if 'Submission_Date' in df.columns:
        df['Claim_Weekday'] = df['Submission_Date'].dt.day_name()
        df['Claim_Month'] = df['Submission_Date'].dt.month_name()
        df['Claim_Quarter'] = df['Submission_Date'].dt.quarter

    df['Claim_Amount_to_Mean'] = df['Claim_Amount_KES'] / df['Claim_Amount_KES'].mean()
    df['Same_Day_Claims'] = df.duplicated(subset=['Employee_ID', 'Submission_Date'], keep=False).astype(int)

    if 'Employer' in df.columns:
        stats = df.groupby('Employer')['Claim_Amount_KES'].agg(['mean', 'std']).reset_index()
        stats.columns = ['Employer', 'Employer_Mean_Claim', 'Employer_Std_Claim']
        df = pd.merge(df, stats, on='Employer', how='left')
        df['Employer_Z_Score'] = (df['Claim_Amount_KES'] - df['Employer_Mean_Claim']) / df['Employer_Std_Claim']

    return df