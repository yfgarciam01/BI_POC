# --- Synthetic Data Generation Script for Google Colab ---
#
# This script loads your 10-table schema, learns the relationships
# and data patterns, and then generates 1,000,000 new, synthetic
# rows for the 'fact_distribution_activity' table.
#
# --- INSTRUCTIONS FOR GOOGLE COLAB ---
# 1. Upload all 10 of your .csv files to your Colab environment.
# 2. Run the first cell in your notebook with these commands:
#    !pip install sdv
#    !pip install sdv-enterprise
# 3. Copy and paste the rest of this script into a new cell and run it.

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.metadata import MultiTableMetadata 
from sdv.multi_table import HMASynthesizer
import warnings
import os
import pickle # Library for saving/loading Python objects

# Suppress common SDV warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
MODEL_FILENAME = 'trained_synthesizer.pkl'

print("--- Synthetic Data Generation Plan ---")
print("STEP 1: Load all 10 tables into memory.")
print("STEP 2: Define the 10-table schema (metadata) and 11 relationships.")
print("STEP 3: 'Learn' the data patterns by training the synthesizer.")
print("STEP 4: Generate 5,000,000 new rows for the fact tables in batches of 500k.")
print("STEP 5: Save each batch to a separate CSV file.")
print("-" * 40)

# --- STEP 1: Load All 10 Tables ---
# Dictionary mapping filenames to their logical table names
FILE_TO_TABLE_MAP = {
    "fact_distribution_activity.csv": "fact_activity",
    "fact_distribution_activity_items.csv": "fact_items",
    "dim_organization.csv": "dim_organization",
    "dim_district.csv": "dim_district",
    "dim_school.csv": "dim_school",
    "dim_user.csv": "dim_user",
    "dim_distribution.csv": "dim_distribution",
    "dim_activity_type.csv": "dim_activity_type",
    "dim_activity_status.csv": "dim_activity_status",
    "dim_vendor.csv": "dim_vendor"
}

path_ = "/home/ygarcia/repos/BI_POC/Data_to_train/"
    
def load_data():

    tables = {}
    print("STEP 1: Loading all 10 datasets...")
    try:
        for f_name, t_name in FILE_TO_TABLE_MAP.items():
            file_path = path_ + f_name
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Error: File not found: {f_name}. Please make sure all 10 CSVs are uploaded.")
            tables[t_name] = pd.read_csv(file_path)
            print(f"-> Loaded '{f_name}' as '{t_name}' ({len(tables[t_name])} rows)")
            
        print("All tables loaded successfully.\n")
        return tables
    except FileNotFoundError as e:
        print(e)
        # Stop execution if files are missing
        exit()
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        exit()

# --- STEP 2: Define the Schema and Relationships (Metadata) ---

def define_metadata(tables):
    
    print("STEP 2: Defining the data schema and relationships...")
    
    data={
            'fact_activity': tables["fact_activity"],
            'fact_items': tables["fact_items"],
            'dim_organization': tables["dim_organization"],
            'dim_district': tables["dim_district"],
            'dim_school': tables["dim_school"],
            'dim_user': tables["dim_user"],
            'dim_distribution': tables["dim_distribution"],
            'dim_activity_type': tables["dim_activity_type"],
            'dim_activity_status': tables["dim_activity_status"],
            'dim_vendor': tables["dim_vendor"]
        }
    
    
    try:
    
        from sdv.metadata import Metadata
        
        metadata = Metadata.load_from_json(filepath=path_ + 'metadata_v1_copy.json')

        # --- Function to Define Metadata (Same as before) ---
        print("Validation results", metadata.validate())  # checks for consistency
        
        # Export data MER
        metadata.visualize(
            show_table_details='full',
            show_relationship_labels=True,
            output_filepath=path_ + 'my_metadata.png'
        )
        
        missing_refs = set(tables['fact_items']['activity_ref_key']) - set(tables['fact_activity']['activity_ref_key'])
        
        print(f"Missing references in fact_items: {len(missing_refs)}")

        from sdv.utils import drop_unknown_references

        cleaned_data = drop_unknown_references(data, metadata)

        return metadata        
        
    except Exception as e:
        print(f"An error occurred during metadata INITIAL relational setup: {e}")
        exit()

def model_trainig(data_tables, model_metadata):
    
    print("\n3. Training the HMASynthesizer (This is the long part)...")
    synthesizer = HMASynthesizer(model_metadata)
    synthesizer.fit(data=data_tables)
    print("-> Training complete.")

    print(f"\n4. Saving the trained model to '{MODEL_FILENAME}'...")
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(synthesizer, f)
    print(f"--- SUCCESS: Model saved as {MODEL_FILENAME} ---")


# --- Main Execution ---
if __name__ == '__main__':
    try:
        data_tables = load_data()
        model_metadata = define_metadata(data_tables)
        
        # model_trainig(data_tables, model_metadata)
        
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")    