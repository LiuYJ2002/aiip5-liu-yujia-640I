from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def preprocess_data(df, config):
    """
    Preprocess data: Correct data types, handle missing values and duplicate rows, standardise categorical values, remove outliers, one hot encode categorical data
    """
    # Correct data types
    df['Nutrient N Sensor (ppm)'] = pd.to_numeric(df['Nutrient N Sensor (ppm)'], errors='coerce')
    df['Nutrient P Sensor (ppm)'] = pd.to_numeric(df['Nutrient P Sensor (ppm)'], errors='coerce')
    df['Nutrient K Sensor (ppm)'] = pd.to_numeric(df['Nutrient K Sensor (ppm)'], errors='coerce')
    # Drop duplicate rows
    if config['duplicates']:
        print("removing duplicates")
        df = df.drop_duplicates()
    # Handle missing values
    numeric_columns = df.select_dtypes(include=['number']).columns
    if config['missing']:
        print("filling missing values")
        df.loc[:, numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    # Standardise categorical values
    df.loc[:, 'Plant Type'] = df['Plant Type'].str.strip().str.lower()
    df.loc[:, 'Plant Stage'] = df['Plant Stage'].str.strip().str.lower()
    df.loc[:, 'Previous Cycle Plant Type'] = df['Previous Cycle Plant Type'].str.strip().str.lower()

    #create new column for task
    df['Plant Type-Stage'] = df['Plant Type'] + '-' + df['Plant Stage']

    #outliers
    if config['outliers']:
        print("removing outliers")
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25) 
            Q3 = df[col].quantile(0.75) 
            IQR = Q3 - Q1  # Interquartile range

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Apply filtering to remove outliers
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # Encode categorical columns 
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    # Normalize numeric features using StandardScaler
    if config['normalise']:
        print("normalising data")
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    print(df.head())
    return df

def split_data(df, target_column, type):
    """
    Split data into train/test sets
    """
    df_copy = df.copy()
    if type == 'regression':
        df_copy.drop(columns=['Plant Type-Stage'], inplace=True)
    else:
        df_copy.drop(columns=['Plant Type'], inplace=True)
        df_copy.drop(columns=['Plant Stage'], inplace=True)
    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
