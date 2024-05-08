### src/load_data.py
import pandas as pd
from src.config import raw_file

def load_data():
    return pd.read_csv(raw_file)

# Example usage:
if __name__ == "__main__":
    data = load_data()