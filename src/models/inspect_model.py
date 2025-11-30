import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# Define the features exactly as they were in the training script
# (In a real project, these should be in a shared config file)
CATEGORICAL_FEATURES = ['city', 'type', 'ownership', 'buildingMaterial', 'condition']
NUMERICAL_FEATURES = [
    'squareMeters', 'rooms', 'floor', 'floorCount', 'buildYear', 
    'centreDistance', 'poiCount', 'schoolDistance', 'clinicDistance', 
    'postOfficeDistance', 'kindergartenDistance', 'restaurantDistance', 
    'collegeDistance', 'pharmacyDistance'
]

def get_feature_names(pipeline):
    """
    Extracts feature names from the ColumnTransformer in the pipeline.
    """
    # 1. Access the preprocessor step
    preprocessor = pipeline.named_steps['preprocessor']
    
    # 2. Get names for numerical features (they remain unchanged)
    # The transformer name 'num' was defined in train_model.py
    # If using StandardScaler, names don't change.
    num_names = NUMERICAL_FEATURES
    
    # 3. Get names for categorical features (after OneHotEncoding)
    # The transformer name 'cat' was defined in train_model.py
    # access the OneHotEncoder object inside the pipeline
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    
    return list(num_names) + list(cat_names)

def plot_importance(mode):
    model_path = f"models/model_{mode}.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        print("Please copy the model from MLflow artifacts to the 'models/' folder first.")
        return

    print(f"Loading {mode} model...")
    pipeline = joblib.load(model_path)
    
    # Extract the Regressor (Random Forest)
    rf_model = pipeline.named_steps['regressor']
    
    # Get Feature Importances
    importances = rf_model.feature_importances_
    
    # Get Feature Names
    feature_names = get_feature_names(pipeline)
    
    # Create DataFrame
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(20) # Top 20
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
    plt.title(f'Top 20 Feature Importances - {mode.upper()} Model')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
    
    # Print the raw data for top 10
    print("\nTop 10 Features:")
    print(feat_df.head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['sale', 'rent'])
    args = parser.parse_args()
    
    plot_importance(args.mode)