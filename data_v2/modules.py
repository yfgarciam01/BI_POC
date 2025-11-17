import math
import os
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)

def export_to_csv_from_df(df, base_path, table_name, batch_size):
    """
    Export a Datafram into multiple csv files by batches
    
    Args:
        df (pd.DataFrame): Dataframe to export
        base_path (str): Output directory
        table_name (str): Prefix for the exported files
        batch_size (int): Number of rows per file
    """
    
    os.makedirs(base_path, exist_ok=True) # Create the directory if it doesn't exist
    n_rows = len(df)
    print(f"n_rows: {n_rows}")
    n_batches = math.ceil(n_rows / batch_size)
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_rows)
        batch_df = df.iloc[start:end]
        file_path = os.path.join(base_path, f"{table_name}_part_{i+1:04d}.csv")
        
        # Define compression options ZIP
        # compression_opts = dict(method='zip', archive_name='my_data.csv')
        batch_df.to_csv(file_path, index=False)
        
        logging.info(f"Saved {file_path} ({len(batch_df)} rows)")
        
        
# --- Main Execution ---
if __name__ == '__main__':
    try:
        
        df_= pd.read_parquet('/home/ygarcia/repos/BI_POC/data_v2/data_gen/synthetic_fact_items.parquet')
        path_ = "/home/ygarcia/repos/BI_POC/data_v2/data_gen/fact_dist_items"
        
        export_to_csv_from_df(df_, path_, 'fact_distribution_activity_items', 400000)
        
    except Exception as e:
        logging.info(f"\nAN UNEXPECTED ERROR: {e}")