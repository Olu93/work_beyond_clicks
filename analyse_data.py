# %%
import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_json('testdata/raw_cds_data_json_2022-04-01_2022-07-04.jsonl', orient='records',  lines=True)
df
# %%
df["enriched_article"].iloc[0]
# %%
