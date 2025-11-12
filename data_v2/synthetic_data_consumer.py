import pandas as pd
import warnings
import os
from sdv.utils import load_synthesizer
import pandas as pd
import numpy as np
import uuid

import logging
logging.basicConfig(level=logging.INFO)

from synthetic_data_generator import load_data, define_metadata

warnings.filterwarnings('ignore', category=UserWarning)

fact_activity_model = 'fact_activity_model.pkl'
fact_activity_items = 'fact_items_model.pkl'
path_ = "/home/ygarcia/repos/BI_POC/data_v2"

logging.info("--- STARTING GENERATION PIPELINE ---")

if not os.path.exists(f"{path_}/data_gen/{fact_activity_model}"):
    raise FileNotFoundError(f"Model file not found at {path_} {fact_activity_model}")

if not os.path.exists(f"{path_}/data_gen/{fact_activity_items}"):
    raise FileNotFoundError(f"Model file not found at {path_} {fact_activity_items}")

synthesizer_activity = load_synthesizer(filepath=f"{path_}/data_gen/{fact_activity_model}")
synthesizer_items = load_synthesizer(filepath=f"{path_}/data_gen/{fact_activity_items}")

logging.info("-> Model loaded successfully.")

# --- Generate synthetic data ---
logging.info("\n --- Generating synthetic data --- \n")

try:
    
    n = 3000000
    
    data_tables = load_data()
    cleaned_data, model_metadata = define_metadata(data_tables)
            
    logging.info("\n # _______________________FACT_ACTIVITY_______________________________________ # \n")
    
    synthetic_data_activity = synthesizer_activity.sample(num_rows=n)
    synthetic_data_activity['activity_ref_key'] = [uuid.uuid4().hex for _ in range(n)]
    
    # Assign random valid FK values from dims
    fk_map = {
        'activity_status_id': 'dim_activity_status',
        'activity_type_id': 'dim_activity_type',
        'distribution_id': 'dim_distribution',
        'district_id': 'dim_district',
        'organization_id': 'dim_organization',
        'school_id': 'dim_school',
        'user_id': 'dim_user',
        'vendor_id': 'dim_vendor',
    }
    
    for fk, dim_name in fk_map.items():
        synthetic_data_activity[fk] = np.random.choice(cleaned_data[dim_name].iloc[:,0], size=n, replace=True)
        #logging.info(f"dim_name: {dim_name} - {synthetic_data_activity[fk]}")
            
    logging.info("# _______________________FACT_ACTIVITY ITEMS_______________________________________ #")
    
    synthetic_data_items = synthesizer_items.sample(n)
    synthetic_data_items['item_id'] = range(1, n+1)

    synthetic_data_items['activity_ref_key'] = np.random.choice(synthetic_data_activity['activity_ref_key'], size=n, replace=True)
    
except Exception as e:
    logging.info(f"Error during data generation: {e}")
    
try:
    
    # Verify consistency between tables
    logging.info(f"Verify consistency between tables FACTS\n")
    # For FACTS
    assert set(synthetic_data_items['activity_ref_key']).issubset(set(synthetic_data_activity['activity_ref_key']))
    logging.info(f"FACTS OK\n")
    
    logging.info(f"Verify consistency between tables DIMS\n")
    # For DIMS
    for fk, dim_name in fk_map.items():
        assert set(synthetic_data_activity[fk],).issubset(set(cleaned_data[dim_name].iloc[:,0]))
    logging.info(f"DIMS OK\n")

except Exception as e:
    logging.info(f"Error during data verification: {e}")
    
    
# Dump into files    
try:
    
    logging.info("\n Data dump into Parquet files:\n")
    
    # --- FACT_ACTIVITY ---
    output_path_act = os.path.join(path_, f"synthetic_fact_activity.parquet")
    synthetic_data_activity.to_parquet(output_path_act, index=False)
    logging.info(f"Saved {len(synthetic_data_activity):,} rows for fact_activity → {output_path_act} \n")
    
    # --- FACT_ACTIVITY ITEMS ---
    output_path_items = os.path.join(path_, f"synthetic_fact_items.parquet")
    synthetic_data_items.to_parquet(output_path_items, index=False)
    logging.info(f"Saved {len(synthetic_data_items):,} rows for fact_activity items → {output_path_items} \n")
    
except Exception as e:
    logging.info(f"Error during data dump: {e}")
       
        
logging.info("\n--- GENERATION COMPLETE ---")
