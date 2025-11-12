import pandas as pd
import warnings
import os
from sdv.utils import load_synthesizer

warnings.filterwarnings('ignore', category=UserWarning)

MODEL_FILENAME = 'fact_activity_model.pkl'
path_ = "/home/ygarcia/repos/BI_POC/data_v2/"


def generate_fact_activity(model,n, dims):
    fact= model.sample(n)
    
    # Assign random valid FK values 

print("--- STARTING GENERATION PIPELINE ---")
print(f"1. Loading model from '{MODEL_FILENAME}'...")

if not os.path.exists(path_ + MODEL_FILENAME):
    raise FileNotFoundError(f"Model file not found at {path_ + MODEL_FILENAME}")

synthesizer = load_synthesizer(filepath=path_ + MODEL_FILENAME)

print("-> Model loaded successfully.")



# --- Generate synthetic data ---
print("\n2. Generating synthetic data...")

try:
    pass
except Exception as e:
    print(f"Error during data generation: {e}")

print("\n--- GENERATION COMPLETE ---")
