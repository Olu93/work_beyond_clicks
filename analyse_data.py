# %%
import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import flatdict
# %%
df = pd.read_json('testdata/raw_cds_data_json_2022-04-01_2022-07-04.jsonl', orient='records',  lines=True)
df
# %%
def transform_level_1(x:pd.DataFrame):
    d = flatdict.FlatDict(x, delimiter='.')    
    return d
def transform_1_to_1_list(x:pd.DataFrame):
    d = flatdict.FlatDict(x[0], delimiter='.')    
    return d
# %%
sub_enriched = df["enriched_article"].apply(transform_level_1).apply(pd.Series)
sub_enriched
# %%
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(sub_enriched.select_dtypes('object').dtypes)
# display(sub_enriched.select_dtypes('object'))
# display(sub_enriched['brand_publications'].apply(len).max())
# %%
sub_images = sub_enriched["images"].apply(transform_1_to_1_list).apply(pd.Series)
sub_brand_publication = sub_enriched["brand_publications"].apply(transform_1_to_1_list).apply(pd.Series)
sub_brand_categories = sub_enriched["categories"].apply(lambda x: "|".join(x))
sub_brand_source_keywords = sub_enriched["source_keywords"].apply(lambda x: "|".join(x))
sub_complex_enrichments = sub_enriched[[col for col in sub_enriched.select_dtypes('object').columns if 'enrichment' in col]].drop(["enrichments.userneeds.max", "enrichments.language"], axis=1)
# %%
# %%


# %%
