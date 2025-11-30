import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# 1. Configuration
# Define which columns are categorical and which are numerical
# specific to your dataset structure
CATEGORICAL_FEATURES = ['city', 'type', 'ownership', 'buildingMaterial', 'condition']
NUMERICAL_FEATURES = [
    'squareMeters', 'rooms', 'floor', 'floorCount', 'buildYear', 
    'centreDistance', 'poiCount', 'schoolDistance', 'clinicDistance', 
    'postOfficeDistance', 'kindergartenDistance', 'restaurantDistance', 
    'collegeDistance', 'pharmacyDistance', 'hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom'
]

def load_data(mode):
    """Load the processed CSV based on mode (rent/sale)"""
    path = f"data/processed/{mode}_structured.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found. Did you run preprocessing?")
    return pd.read_csv(path)

def build_pipeline():
    """Create a Scikit-learn pipeline with preprocessing and model"""
    
    # Preprocessing for numerical data: Scale it
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data: OneHotEncode it
    # handle_unknown='ignore' is crucial for production! 
    # If a new city appears that wasn't in training, it won't crash.
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numeric and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])

    # Add the model (Random Forest for now)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    return model

def main(mode):
    # Set the experiment name in MLflow
    mlflow.set_experiment(f"housing_prices_{mode}")
    
    print(f"Loading {mode} data...")
    df = load_data(mode)
    
    # Separate Target and Features
    X = df.drop(columns=['price', 'report_date']) # Drop date for now to keep it simple
    y = df['price']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Starting training...")
    
    # Start MLflow Run
    with mlflow.start_run():
        # Build and Train Pipeline
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        
        # Predict
        predictions = pipeline.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)
        
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print(f"R2: {r2}")
        
        # Log Parameters and Metrics to MLflow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("data_mode", mode)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Log the complete pipeline model
        mlflow.sklearn.log_model(pipeline, "model")
        
        print("Model saved to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['sale', 'rent'], help="Choose 'sale' or 'rent'")
    args = parser.parse_args()
    main(args.mode)