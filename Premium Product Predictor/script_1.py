"""
Premium product predictor:
Build a model which can predict whether a product is premium or non-premium.

Disclaimer:
Due to confidentiality reasons, it is not possible to present or divulge all aspects of the dataset used in this
project.
"""
# %%
# libraries
import pyarrow.parquet as pq
import numpy as np

# %%
# reading data
table = pq.read_table('.../premium-product-predictor/dataset.parquet')
df = table.to_pandas()

# %%
# checking columns names
df.columns


# %%
# reordering columns
column_names_reordered = ['column_33', 'column_72', 'column_35', 'column_58', 'column_56',
                          'column_27', 'column_73', 'column_64', 'column_22', 'column_16',
                          'column_71', 'column_99', 'column_52', 'column_74', 'column_41',
                          'column_79', 'column_65', 'column_48', 'column_94', 'column_24',
                          'column_98', 'column_81', 'column_36', 'column_23', 'column_75',
                          'column_63', 'column_25', 'column_60', 'column_46', 'column_80',
                          'column_82', 'column_45', 'column_50', 'column_17', 'column_68',
                          'column_40', 'column_19']
df = df[column_names_reordered]

# %%
# checking for null values
df.isna().sum()

# %%
# checking for duplicated values
df.duplicated().sum()

# %%
# getting unique values for each variable
for col in df:
    print(col, '=', df[col].unique())

# %%
# changing data type from float to int
df['column_33'] = df['column_33'].astype(np.int64)
df['column_72'] = df['column_72'].astype(np.int64)
df['column_35'] = df['column_35'].astype(np.int64)
df['column_27'] = df['column_27'].astype(np.int64)
df['column_56'] = df['column_56'].astype(np.int64)
df['column_73'] = df['column_73'].astype(np.int64)
df['column_58'] = df['column_58'].astype(np.int64)

# %%
# drop column_35 and column_72
# column_35 generates data leakage
# column_72 is not relevant according to the domain experts
df = df.drop(['column_35', 'column_72'], axis=1)

# saving to a CSV file
df.to_csv('.../premium-product-predictor/dataset.csv', index=False)
