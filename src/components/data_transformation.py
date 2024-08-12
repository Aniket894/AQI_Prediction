# src/components/data_transformation.py
import pandas as pd
import numpy as np
import joblib
from src.logger import logging
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def transform_data():
    logging.info("Started data transformation")
    
    try:
        # Load train and test datasets
        train_data = pd.read_csv('artifacts/train.csv')
        test_data = pd.read_csv('artifacts/test.csv')
        logging.info("Loaded train and test data")

        # Convert 'Date' to datetime for train and test datasets
        train_data['Date'] = pd.to_datetime(train_data['Date'])
        test_data['Date'] = pd.to_datetime(test_data['Date'])

        # Extract Year, Month, and Day from 'Date'
        train_data['Year'] = train_data['Date'].dt.year
        train_data['Month'] = train_data['Date'].dt.month
        train_data['Day'] = train_data['Date'].dt.day
        
        test_data['Year'] = test_data['Date'].dt.year
        test_data['Month'] = test_data['Date'].dt.month
        test_data['Day'] = test_data['Date'].dt.day

        # Separate features and target variable
        target = 'AQI'
        cols_to_drop = ['Site', 'State', 'Date']
        X_train = train_data.drop(columns=[target] + cols_to_drop)
        y_train = train_data[target]
        X_test = test_data.drop(columns=[target] + cols_to_drop)
        y_test = test_data[target]

        # Identify numerical and categorical columns
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X_train.select_dtypes(include=['object']).columns

        # Define transformations for numerical and categorical columns
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal_encoder', OrdinalEncoder(handle_unknown='ignore'))
        ])

        # Combine transformers into a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Fit and transform the data
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        logging.info("Transformed train and test data")

        # Save the transformed datasets and preprocessor
        np.save('artifacts/X_train.npy', X_train_transformed)
        np.save('artifacts/y_train.npy', y_train.values)  # Save y_train and y_test as arrays
        np.save('artifacts/X_test.npy', X_test_transformed)
        np.save('artifacts/y_test.npy', y_test.values)
        joblib.dump(preprocessor, 'artifacts/preprocessor.pkl')
        logging.info("Saved transformed datasets and preprocessor to artifacts folder")

    except Exception as e:
        logging.error(f"An error occurred during data transformation: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    transform_data()
