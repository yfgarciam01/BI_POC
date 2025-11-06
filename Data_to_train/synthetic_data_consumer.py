# --- GENERATION PIPELINE: Load Model and Generate Batches ---
#
# This script is designed to be run *after* the training_pipeline.py
# It loads the pre-trained model and generates the 5M rows in batches
# without needing to retrain.
#
# REQUIRED: 'trained_synthesizer.pkl' file must be present.

import pandas as pd
import warnings
import pickle
import os

warnings.filterwarnings('ignore', category=UserWarning)

MODEL_FILENAME = 'trained_synthesizer.pkl'
TOTAL_ROWS_TO_GENERATE = 5_000_000
BATCH_SIZE = 500_000
num_batches = TOTAL_ROWS_TO_GENERATE // BATCH_SIZE

print("--- STARTING GENERATION PIPELINE ---")

# --- Step 1: Load the Trained Model ---
print(f"1. Attempting to load pre-trained model from '{MODEL_FILENAME}'...")

if not os.path.exists(MODEL_FILENAME):
    print(f"Error: Model file '{MODEL_FILENAME}' not found.")
    print("Please run 'training_pipeline.py' first to create the trained model.")
    exit()

try:
    with open(MODEL_FILENAME, 'rb') as f:
        synthesizer = pickle.load(f)
    print("-> Model loaded successfully. Ready for generation.")
except Exception as e:
    print(f"Error loading the pickled model: {e}")
    exit()


# --- Step 2: Generate and Save in Batches ---
print(f"\n2. Generating {TOTAL_ROWS_TO_GENERATE:,} rows in {num_batches} batches of {BATCH_SIZE:,}...")

for i in range(num_batches):
    batch_num = i + 1
    print(f"\n--- Generating Batch {batch_num} of {num_batches} ({BATCH_SIZE:,} rows) ---")

    try:
        # Sample the fact_activity table; SDV generates associated fact_items rows
        new_synthetic_data = synthesizer.sample_table(
            table_name='fact_activity',
            num_rows=BATCH_SIZE
        )

        synthetic_fact_activity = new_synthetic_data['fact_activity']
        synthetic_fact_items = new_synthetic_data['fact_items']

        # Define output filenames for this batch
        OUTPUT_FILE_ACTIVITY = f"synthetic_fact_activity_batch_{batch_num}.csv"
        OUTPUT_FILE_ITEMS = f"synthetic_fact_items_batch_{batch_num}.csv"

        # Save the activity fact table
        print(f"-> Saving activity data to '{OUTPUT_FILE_ACTIVITY}'...")
        synthetic_fact_activity.to_csv(OUTPUT_FILE_ACTIVITY, index=False)

        # Save the items fact table
        print(f"-> Saving items data to '{OUTPUT_FILE_ITEMS}'...")
        synthetic_fact_items.to_csv(OUTPUT_FILE_ITEMS, index=False)

        print(f"-> Batch {batch_num} complete. Saved {len(synthetic_fact_activity):,} rows.")

    except Exception as e:
        print(f"An error occurred during batch {batch_num} generation or saving: {e}")
        break

print("\n--- GENERATION COMPLETE ---")
print(f"Total of {TOTAL_ROWS_TO_GENERATE:,} rows processed across 10 batches.")