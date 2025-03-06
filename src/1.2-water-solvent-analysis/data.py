import sqlite3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class SolventAnalysis:
    def __init__(self, query):
        self.query = query
        self.scaler = MinMaxScaler()
        self.df = self._load_data()
        
    def _load_data(self):
        conn = sqlite3.connect('db/MasterDatabase.db')
        
        df = pd.read_sql_query(self.query, conn)
        
        distinct_apis = list(map(int, df['api'].unique()))
        distinct_solvents = list(map(int, df['solvent'].unique()))
        
        api_ids = ', '.join(map(str, distinct_apis))
        api_query = f"SELECT * FROM compounds WHERE id IN ({api_ids});"
        api_df = pd.read_sql_query(api_query, conn)
        
        solvent_ids = ', '.join(map(str, distinct_solvents))
        solvent_query = f"SELECT * FROM solvents WHERE id IN ({solvent_ids});"
        solvent_df = pd.read_sql_query(solvent_query, conn)
        conn.close()
        
        api_df = api_df.iloc[:, 3:]
        solvent_df = solvent_df.iloc[:, 3:]
        
        df = df.merge(api_df, left_on='api', right_on='id', how='inner').drop(columns=['id'])
        df = df.merge(solvent_df, left_on='solvent', right_on='id', how='inner').drop(columns=['id'])
        
        df.fillna(0, inplace=True)
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        df.drop(cols_to_drop, axis=1, inplace=True)
        
        columns_to_normalize = df.columns.difference(['api', 'solvent', 'fraction'])
        df[columns_to_normalize] = self.scaler.fit_transform(df[columns_to_normalize])
        
        return df
    
    def inverse_transform(self, data):
        columns_to_normalize = self.df.columns.difference(['api', 'solvent', 'fraction'])
        return self.scaler.inverse_transform(data[columns_to_normalize])
    
    def get_data(self):
        return self.df
    

query = """
SELECT 
    compound_id AS api,
    CASE 
        WHEN solvent_1 = 962 THEN solvent_2
        ELSE solvent_1
    END AS solvent,
    ROUND(solubility_g_g, 4) as solubility,
    CASE 
        WHEN solvent_1 = 962 THEN 1 - ROUND(solvent_1_weight_fraction, 2)
        ELSE ROUND(solvent_1_weight_fraction, 2)
    END AS fraction,
    ROUND(temperature) as temperature 
FROM
    solubility s
WHERE 
    (solvent_2 = 962 OR solvent_1 = 962)
    AND solvent_2 IS NOT NULL
    AND solubility <> 0
    AND fraction <> 1
    AND fraction <> 0;
"""
water = SolventAnalysis(query)

if __name__ == '__main__':
    print(water.df)