import pandas as pd
import glob
import os

def load_and_combine_data(folder_path, dataset_type):
    """
    Loads all CSV files from a specific folder, adds a date column based on filename,
    and merges them into a single DataFrame.
    
    Args:
        folder_path (str): Path to the folder containing monthly CSVs.
        dataset_type (str): 'sale' or 'rent' (used for logging).
        
    Returns:
        pd.DataFrame: The combined dataframe.
    """
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []
    
    print(f"Found {len(all_files)} files for {dataset_type}...")

    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            
            # Extract date from filename (assuming format: apartments_pl_2023_09.csv)
            # We take the last two parts usually: YYYY_MM
            basename = os.path.basename(filename)
            # distinct parsing based on known file patterns
            # Sale pattern: apartments_pl_2023_09.csv
            # Rent pattern: apartments_rent_pl_2023_11.csv
            
            parts = basename.replace('.csv', '').split('_')
            year = parts[-2]
            month = parts[-1]
            
            # Create a simple date string or datetime object
            df['report_date'] = pd.to_datetime(f"{year}-{month}-01")
            
            df_list.append(df)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not df_list:
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows for {dataset_type}: {len(combined_df)}")
    return combined_df

def clean_data(df):
    """
    Applies basic cleaning: mapping booleans, handling types.
    """
    # 1. Map yes/no to 1/0
    boolean_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 
                    'hasSecurity', 'hasStorageRoom']
    
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    # 2. Handle missing 'condition'
    # For a thesis, it's safer to label missing as 'unknown' than to drop rows 
    # or guess the condition.
    if 'condition' in df.columns:
        df['condition'] = df['condition'].fillna('unknown')

    # 3. Drop duplicates if any (same offer appearing in consecutive months)
    # Note: In real estate, an offer might stay up for months. 
    # For this simplified project, we might keep them to show market prevalence, 
    # or drop specific duplicates. Let's drop exact row duplicates.
    df = df.drop_duplicates()

    # 4. Drop rows where target 'price' is missing
    df = df.dropna(subset=['price'])
    
    return df

def main():
    # Define paths
    raw_sale_path = "data/raw/apartments"
    raw_rent_path = "data/raw/rent"
    processed_path = "data/processed"
    
    os.makedirs(processed_path, exist_ok=True)

    # --- PROCESS SALE DATA ---
    print("Processing Sale Data...")
    df_sale = load_and_combine_data(raw_sale_path, "sale")
    if not df_sale.empty:
        df_sale = clean_data(df_sale)
        save_path = os.path.join(processed_path, "sale_structured.csv")
        df_sale.to_csv(save_path, index=False)
        print(f"Saved Sale data to {save_path}")

    # --- PROCESS RENT DATA ---
    print("-" * 30)
    print("Processing Rent Data...")
    df_rent = load_and_combine_data(raw_rent_path, "rent")
    if not df_rent.empty:
        df_rent = clean_data(df_rent)
        save_path = os.path.join(processed_path, "rent_structured.csv")
        df_rent.to_csv(save_path, index=False)
        print(f"Saved Rent data to {save_path}")

if __name__ == "__main__":
    main()