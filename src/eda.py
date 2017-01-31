import numpy as np
import pandas as pd
import os

def process_data(file_name):
    file_path = os.path.join('data', file_name)
    df = pd.read_csv(file_path)
    return df

if __name__ == '__main__':
    import eda
