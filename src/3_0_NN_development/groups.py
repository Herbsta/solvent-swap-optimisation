import sqlite3
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import warnings
import os
import glob
import argparse
import scipy.stats as stats

random.seed(123)

# Database connection and data loading
current_folder = os.path.dirname(os.path.abspath(__file__))
connection = sqlite3.connect(f'{current_folder}/../../db/MasterDatabase.db')
df = pd.read_sql_query("SELECT * FROM selected_solubility_data", connection)
connection.close()

# Groups for JA model (grouped by temperature)
ja_groups = [group.reset_index(drop=True) for _, group in df.groupby(['solvent_1', 'solvent_2', 'compound_id', 'temperature'])]

# Groups for JAVH model (no temperature grouping)
javh_groups = [group.reset_index(drop=True) for _, group in df.groupby(['solvent_1', 'solvent_2', 'compound_id'])]