# %%
import pathlib
from typing import List
import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import flatdict
import numpy as np
from dotenv import load_dotenv
import json
import os
import boto3
import sys
import threading
from io import BytesIO, StringIO, TextIOWrapper
import gzip
import collections
import tqdm
import io
from benedict import benedict

# %%
load_dotenv()  # take environment variables from .env.

# %%
# prerequisite: set 2 env variables, ACCESS_KEY and SECRET_KEY, to the user credentials that are available in LastPass

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ["ACCESS_KEY"],
    aws_secret_access_key=os.environ["SECRET_KEY"],
)

BUCKET_NAME = "recosearch-nps-bucket-shared-with-utrecht-university"
FILE_NAME = "20220909-test-on-tracking-event-file-partition-and-compression/tracking_events_single_file_compressed_json_2022-06-30/part-00000-d667af8c-fe5d-4c75-b144-400a5560425d-c000.json.gz"

# list all directories and files inside the bucket
s3_objects = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
# file_list = [pathlib.Path(item["Key"]) for item in s3_objects["Contents"] if "json.gz" in item["Key"] and "single_file" in item["Key"]]
file_list = [pathlib.Path(item["Key"]) for item in s3_objects["Contents"] if "cds" in item["Key"] and ".json" in item["Key"] and (not "test-on-file" in item["Key"])]
print(file_list)

# %%
import csv
counter = collections.defaultdict(int)
raw_collector = collections.deque()
view_collector = collections.deque()
interaction_collector = collections.deque()
geo_mapping = collections.defaultdict(str)
city_mapping = collections.defaultdict(lambda: len(city_mapping)+1)
limit = 10000
update_freq = 5


def filter_article(tmp_dict):
    if tmp_dict['new_article'] == 'True':
        return tmp_dict
    return None
    
    
def reduce_article(orig_dict):
    tmp_dict = orig_dict["enriched_article"]
    new_dict = {}
    new_dict['article_id'] = orig_dict['article_id']
    new_dict['cds_content_id'] = orig_dict['cds_content_id']
    new_dict['brands'] = tmp_dict['brand']
    new_dict['title'] = tmp_dict['title']
    new_dict['text'] = tmp_dict['text_cleaned']
    new_dict['authors'] = tmp_dict['authors']
    new_dict['url'] = tmp_dict.get_str('url')
    new_dict['main_section'] = tmp_dict['section.main_section']
    new_dict['sub_section'] = tmp_dict.get_str('section.sub_section')
    new_dict['num_words'] = tmp_dict.get_int('enrichments.num_words')
    new_dict['num_sentences'] = tmp_dict.get_int('enrichments.num_sentences')
    new_dict['num_chars'] = tmp_dict.get_int('enrichments.raw_character_count')
    new_dict['first_publication_timestamp'] = tmp_dict.get_datetime('first_publication_ts')
    new_dict['categories_generated'] = tmp_dict.get_str_list('categories')
    new_dict['categories_curated'] = tmp_dict.get_str_list('source_keywords')
    new_dict['categories_textrazor'] = [dict(score=d['score'], label=d['label']) for  d in tmp_dict.get_list('enrichments.categories')]
    new_dict['topics_curated'] = [dict(score=d['score'], name=d['label']) for  d in tmp_dict.get_list('enrichments.topics')]
    new_dict['topics_generated'] = tmp_dict.get_dict('enrichments.ci_topics_v2')
    new_dict['brand_safety'] = tmp_dict.get_dict('enrichments.brand_safety')
    
    return new_dict



for file_name in tqdm.tqdm(file_list[:2], desc="File"):
    gzipfile = s3_client.get_object(Bucket=BUCKET_NAME, Key=str(file_name))["Body"]
    content = TextIOWrapper(gzipfile)
    for cnt, l in tqdm.tqdm(enumerate(content), desc="Lines"):
        tmp_dict = benedict(l)
        raw_collector.append(tmp_dict.copy())
        tmp_dict = reduce_article(tmp_dict)
        view_collector.append(tmp_dict)
        # if ((cnt+1)%update_freq) == 0:
        #     pbar.update(update_freq)
        # if cnt > limit:
        #     break


df_views = pd.DataFrame(view_collector)
df_interactions = pd.DataFrame(interaction_collector)
df_raw = pd.DataFrame(raw_collector)
# df_views.to_csv("data_dpg_testdata/reduced_views2.csv", index=None)
# df_interactions.to_csv("data_dpg_testdata/reduced_interactions2.csv", index=None)
# df_raw.to_csv("data_dpg_testdata/raw2.csv", index=None)

# out = io.open("data_dpg_testdata/reduced_views.csv.gz", "wb")
# out = io.StringIO(df_views.to_csv(index=None))
# with gzip.GzipFile(fileobj=out, mode="w") as f:
#     # f.write(df_views.to_csv(index=None))
#     f.write(out)
data = bytes(df_views.to_csv(index=None), encoding="utf-8")
s_out = gzip.compress(data)
io.open("data_dpg_testdata/reduced_views.csv.gz", mode="wb").write(s_out)
# %%
for col in df_views.columns:
    print(f"---------- Column {col} -----------")
    display(df_views[col].value_counts())

# %%
# SE_ACTION: Kinda like social engagement
# PRIVACYWALL_ID: Only for web - Best identifier if GIGA_ID not available
# GIGYA_ID: Only logged in user
# DOMAIN_USERID: Not used - Hence, we do not need it
# DOMAIN_SESSIONID: Session id does not stay for a long time window. Rosa provides doc of snow-plow

# screen_view from app
# page_view from web
# event if none of the above but then it should have SE_??? information


# Documentation
# - Link: Documentation between timestamps
# - Link: Documentation about difference between DOMAIN_???
