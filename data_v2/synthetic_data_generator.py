# --- Synthetic Data Generation Script ---
#
# This script loads your 10-table schema, learns the relationships
# and data patterns, and then generates millions new, synthetic
# rows for the 'fact_distribution_activity' and 'fact_distribution_activity_items' table.
# Could be used for more tables, the metadata is required previously

import pandas as pd
#from sdv.metadata import SingleTableMetadata
from sdv.metadata import MultiTableMetadata 
from sdv.multi_table import HMASynthesizer
import warnings
import os
import pickle

import logging
logging.basicConfig(level=logging.INFO)


warnings.filterwarnings('ignore', category=UserWarning)

logging.info("--- Synthetic Data Generation Plan ---")
logging.info("STEP 1: Load FACT tables into memory.")
logging.info("STEP 2: Define the 10-table schema (metadata) and 11 relationships.")
logging.info("STEP 3: 'Learn' the data patterns by training the synthesizer.")
logging.info("STEP 4: Save each batch to a separate CSV file.")

# --- STEP 1: Load Tables ---
# Dictionary mapping filenames to their logical table names
FILE_TO_TABLE_MAP = {
    "fact_distribution_activity.csv": "fact_activity",
    "fact_distribution_activity_items.csv": "fact_items",
    "dim_vendor.csv": "dim_vendor",
    "dim_user.csv": "dim_user",
    "dim_school.csv": "dim_school",
    "dim_organization.csv": "dim_organization",
    "dim_district.csv": "dim_district",
    "dim_distribution.csv": "dim_distribution",
    "dim_activity_type.csv": "dim_activity_type",
    "dim_activity_status.csv": "dim_activity_status"
}

path_ = "/home/ygarcia/repos/BI_POC/data_v2"
    
def load_data():

    tables = {}
    
    logging.info("Loading all datasets...")
    
    try:
        for f_name, t_name in FILE_TO_TABLE_MAP.items():
            file_path = f"{path_}/sample_data/{f_name}"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Error: File not found: {f_name}. Please make sure CSVs are uploaded.")
            tables[t_name] = pd.read_csv(file_path)
            logging.info(f"-> Loaded '{f_name}' as '{t_name}' ({len(tables[t_name])} rows)")
            
        logging.info("All tables loaded successfully.\n")
        return tables
    
    except FileNotFoundError as e:
        logging.info(e)
        exit()
    except Exception as e:
        logging.info(f"An error occurred during file loading: {e}")
        exit()

# --- STEP 2: Define the Schema and Relationships (Metadata) ---
def define_metadata(tables):
    
    logging.info("Defining the data schema and relationships.")
    
    data={
            'fact_activity': tables["fact_activity"],
            'fact_items': tables["fact_items"],
            'dim_vendor': tables["dim_vendor"],
            'dim_user': tables["dim_user"],
            'dim_school': tables["dim_school"],
            'dim_organization': tables["dim_organization"],
            'dim_district': tables["dim_district"],
            'dim_distribution': tables["dim_distribution"],
            'dim_activity_type': tables["dim_activity_type"],
            'dim_activity_status': tables["dim_activity_status"]
        }
    
    
    try:
    
        from sdv.metadata import Metadata

        metadata = Metadata.load_from_json(filepath=f"{path_}/metadata_v1.json")

        logging.info("Validation results", metadata.validate())  # checks for consistency
        
        # Export data MER
        metadata.visualize(
            show_table_details='full',
            show_relationship_labels=True,
            output_filepath=f"{path_}/my_metadata.png"
        )
        
        missing_refs = set(tables['fact_items']['activity_ref_key']) - set(tables['fact_activity']['activity_ref_key'])
        
        logging.info(f"Missing references in fact_items: {len(missing_refs)}")

        from sdv.utils import drop_unknown_references

        cleaned_data = drop_unknown_references(data, metadata)
        
        logging.info("\n \n ")

        return cleaned_data, metadata        
        
    except Exception as e:
        logging.info(f"An error occurred during metadata INITIAL relational setup: {e}")
        exit()

# SETP 3 - Model training
def model_trainig(cleaned_data, model_metadata):
    
    logging.info("\nTraining the GaussianCopulaSynthesizer")
    
    #synthesizer = HMASynthesizer(model_metadata)
    #synthesizer.fit(data=cleaned_data)
    
    fact_activity_m = 'fact_activity_model.pkl'
    fact_items_m = 'fact_items_model.pkl'
    
    from sdv.single_table import GaussianCopulaSynthesizer
    
    # fact_activity_model = GaussianCopulaSynthesizer(model_metadata.tables['fact_activity'])
    # fact_activity_model.fit(cleaned_data['fact_activity'])
    
    fact_items_model = GaussianCopulaSynthesizer(model_metadata.tables['fact_items'])
    fact_items_model.fit(cleaned_data['fact_items'])
    
    logging.info("-> Training complete.")

    # STEP 4 saving data
    
    # logging.info(f"\nSaving the trained model to '{fact_activity_m}'...")
    # with open(f"{path_}/data_gen/fact_activity_model.pkl", 'wb') as f:
    #    pickle.dump(fact_activity_model, f)

    logging.info(f"\nSaving the trained model to '{fact_items_m}'...")
    with open(f"{path_}/data_gen/fact_items_model.pkl", 'wb') as f:
        pickle.dump(fact_items_model, f)
        
# --- Main Execution ---
if __name__ == '__main__':
    try:
        data_tables = load_data()
        
        cleaned_data, model_metadata = define_metadata(data_tables)

        # dictionary comprehension
        # metadata_subset_keys = ["fact_activity", "fact_items","dim_organization", "dim_user"]
        
        # subset_cleaned_data = {
        #    key: cleaned_data[key] 
        #    for key in metadata_subset_keys 
        #    if key in cleaned_data
        #}

        logging.info(f"\n \n subset_cleaned_data: {cleaned_data.keys()}")
        
        model_trainig(cleaned_data, model_metadata)
        
    except Exception as e:
        logging.info(f"\nAN UNEXPECTED ERROR: {e}")    