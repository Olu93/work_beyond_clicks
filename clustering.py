# %%
# https://examples.dask.org/machine-learning.html
# https://ml.dask.org/
# https://medium.com/analytics-vidhya/dask-for-python-and-machine-learning-dbe1356b5d7a
# https://www.manifold.ai/dask-and-machine-learning-preprocessing-tutorial
import pandas as pd
import hvplot.pandas
import dask
import hvplot.dask
import dask.dataframe as dd
import json
import tqdm
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg
import panel as pn
pn.extension(comms="vscode")

# %%
print("Load Data")
DATA_FOLDER = "preflight"
ddf_views = dd.read_csv(f'data_dpg_testdata/{DATA_FOLDER}/reduced_views.csv', dtype={'GEO_ZIPCODE': object,
        'REFR_MEDIUM': object, 
        'IS_LOGGED_IN': object, 
        'file_name': object,
        "privacy_advertising":int,       
        "privacy_analytics":int,       
        "privacy_functional":int,       
        "privacy_geo_location":int,       
        "privacy_marketing":int,       
        "privacy_non-personalised_ads":int,       
        "privacy_personalisation":int,       
        "privacy_social_media":int,       
        "privacy_target_advertising":int,       
       })
ddf_views["DERIVED_TSTAMP"] = dd.to_datetime(ddf_views["DERIVED_TSTAMP"])
ddf_views["hour"] = ddf_views["DERIVED_TSTAMP"].dt.hour
ddf_views["weekday"] = ddf_views["DERIVED_TSTAMP"].dt.weekday
ddf_views["dayofmonth"] = ddf_views["DERIVED_TSTAMP"].dt.day
ddf_views["month"] = ddf_views["DERIVED_TSTAMP"].dt.month
ddf_views = ddf_views.replace("nu.web.advertorial", "nu.web")
device_counts = ddf_views["APP_ID"].value_counts().compute()
sample_amount = device_counts.min()
unique_devices = device_counts.index.unique()
columns_privacy = list(ddf_views.columns[ddf_views.columns.str.startswith("privacy_")].values)
columns_time = ["hour", "weekday", "dayofmonth", "month"]
ddf_views = ddf_views.groupby('APP_ID').apply(lambda df: df.sample(sample_amount), meta=ddf_views.partitions[0]).reset_index(drop=True)
ddf_views.head().T

# %%
DATA_FOLDER_ARTICLES = "reduced"
ddf_articles = dd.read_csv(f'data_dpg_testdata/{DATA_FOLDER_ARTICLES}/reduced_articles_df_article.csv', dtype={'sub_section': 'object', 'url': 'object'})
ddf_articles.head().T

# %%
ddf_userneeds = dd.read_csv(f'data_dpg_testdata/{DATA_FOLDER_ARTICLES}/reduced_articles_df_fixed_set_userneeds.csv', dtype={'sub_section': 'object','url': 'object'})
columns_userneeds = list(ddf_userneeds.columns[ddf_userneeds.columns.str.startswith("userneed_")].values)
ddf_userneeds = ddf_userneeds.assign(argmax_userneeds = ddf_userneeds[columns_userneeds].idxmax(axis=1))
ddf_userneeds.head().T

# %%
print("Running Merge")
merge_cols_view = ["ARTICLE_ID", "APP_ID", "QUASI_USER_ID", "IS_LOGGED_IN", "GEO_REGION"]+columns_privacy+columns_time
merge_cols_article = ["article_id", "main_section", "sub_section"]
merge_cols_drop = ['argmax_userneeds', 'file_name']
ddf_merged = ddf_userneeds.drop(merge_cols_drop, axis=1)
ddf_merged = ddf_merged.merge(ddf_views[merge_cols_view], how="inner", left_on='article_id', right_on='ARTICLE_ID')
ddf_merged = ddf_merged.merge(ddf_articles[merge_cols_article], how="inner", left_on='article_id', right_on='article_id')
ddf_merged.head().T

# %%
ddf_merged.head().T

# %%
from dask_ml.compose import ColumnTransformer
from dask_ml.preprocessing import Categorizer, DummyEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from dask_ml.cluster import KMeans
from dask_ml.decomposition import PCA, IncrementalPCA

col_categoricals = ["GEO_REGION", "APP_ID", "main_section", "sub_section", "IS_LOGGED_IN"]
col_numericals = columns_userneeds + columns_time 
col_passed = columns_privacy 
col_dropped = ["ARTICLE_ID", "article_id", "QUASI_USER_ID"] 
prep_categoricals = make_pipeline(Categorizer(), DummyEncoder(drop_first=True), verbose=True)
prep_numericals = StandardScaler()

dim_reducer = PCA(n_components=10, svd_solver='auto')


col_transformer = ColumnTransformer([
    ("cat", prep_categoricals, col_categoricals),
    ("num", prep_numericals, col_numericals),
    ("dropped", 'drop', col_dropped),
    ("unchanged", 'passthrough', col_passed),
], n_jobs=4, preserve_dataframe=True, remainder='drop')
# col_transformer_1 = ColumnTransformer([
#     ("cat", prep_categoricals, col_categoricals),
#     # ("num", prep_numericals, col_numericals),
#     # ("dropped", 'drop', col_dropped),
#     # ("unchanged", 'passthrough', col_passed),
# ], n_jobs=4, preserve_dataframe=True, remainder='passthrough')
# col_transformer_2 = ColumnTransformer([
#     # ("cat", prep_categoricals, col_categoricals),
#     ("num", prep_numericals, col_numericals),
#     # ("dropped", 'drop', col_dropped),
#     # ("unchanged", 'passthrough', col_passed),
# ], n_jobs=4, preserve_dataframe=True, remainder='passthrough')
# col_transformer_3 = ColumnTransformer([
#     # ("cat", prep_categoricals, col_categoricals),
#     # ("num", prep_numericals, col_numericals),
#     # ("dropped", 'drop', col_dropped),
#     ("unchanged", 'passthrough', col_passed),
# ], n_jobs=4, preserve_dataframe=True, remainder='passthrough')
# estimators = [('reduce_dim', PCA(3)), ('cluster', KMeans())]

print("Running Clustering")
print("Step Transform")
ddf_transformed = col_transformer.fit_transform(ddf_merged)
print("Step Dimension Reduction")
ddf_reduced = dim_reducer.fit_transform(ddf_transformed).compute_chunk_sizes() #.compute_chunksizes(lengths=True)
print(ddf_transformed)
print("Step Clustering")
cluster_alg = KMeans().fit(ddf_reduced)
print("Done")

# pipe = make_pipeline(
#     col_transformer, 
#     # col_transformer_1, 
#     # col_transformer_2, 
#     # col_transformer_3, 
#     IncrementalPCA(n_components=10, svd_solver='auto'), 
#     # KMeans(),
#     verbose=True,
# )
# pipe = pipe.fit(ddf_merged)
# pipe

# # %%
# ddf_merged

# # %%
# results = pipe.transform(ddf_merged)
# results

# # %%
# ddf_merged.to_dask_array(lengths=True).compute()

# # %%
# ddf_merged


