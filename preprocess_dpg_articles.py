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
import pyexcel as p
import csv

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
time_filter = "20221114"

file_list = [pathlib.Path(item["Key"]) for item in s3_objects["Contents"] if "cds" in item["Key"] and ".json" in item["Key"] and (time_filter in item["Key"])]
print(file_list)

# %%
mapping_files = {str(f):i for i, f in enumerate(file_list)}
mapping_files
json.dump(mapping_files,io.open("./data_dpg_testdata/reduced/mapping_article_files.json", "w"))
# %%

counter = collections.defaultdict(int)
raw_collector = collections.deque()
collector_article = collections.deque()
interaction_collector = collections.deque()

# Internal data is probably defined by DPG. The values are not fixed or limited.
collector_internal_topics = collections.deque()

# External data is probably defined by another model. The values are not fixed or limited.
collector_external_topics = collections.deque()
collector_external_categories = collections.deque()
collector_external_entities = collections.deque()

# The extend of variable values in this data is fixed.
collector_fixed_set_userneeds = collections.deque()
collector_fixed_set_topics = collections.deque()
collector_fixed_set_sensitive = collections.deque()

geo_mapping = collections.defaultdict(str)
city_mapping = collections.defaultdict(lambda: len(city_mapping) + 1)
limit = 10000
update_freq = 5


def filter_article(tmp_dict):
    if tmp_dict['new_article'] == 'True':
        return True
    return None


def reduce_article(fname, orig_dict):
    tmp_dict = orig_dict["enriched_article"]
    new_dict = {}
    new_dict['article_id'] = orig_dict['article_id']
    new_dict['cds_content_id'] = orig_dict['cds_content_id']
    new_dict['brands'] = tmp_dict['brand']
    new_dict['title'] = tmp_dict['title']
    new_dict['text'] = tmp_dict['text_cleaned']
    new_dict['authors'] = "|".join(tmp_dict.get_str_list('authors'))
    new_dict['url'] = tmp_dict.get_str('url')
    new_dict['main_section'] = tmp_dict['section.main_section']
    new_dict['sub_section'] = tmp_dict.get_str('section.sub_section')
    new_dict['num_words'] = tmp_dict.get_int('enrichments.num_words')
    new_dict['num_sentences'] = tmp_dict.get_int('enrichments.num_sentences')
    new_dict['num_chars'] = tmp_dict.get_int('enrichments.raw_character_count')
    new_dict['first_publication_timestamp'] = tmp_dict.get_datetime('first_publication_ts')
    new_dict['categories_generated'] = "|".join(tmp_dict.get_str_list('categories'))
    new_dict['keywords_curated'] = "|".join(tmp_dict.get_str_list('source_keywords'))
    # new_dict['categories_textrazor'] = [dict(score=d['score'], label=d['label']) for  d in tmp_dict.get_list('enrichments.categories')]
    # new_dict['topics_generated'] = tmp_dict.get_dict('enrichments.ci_topics_v2')
    # new_dict['topics_curated'] = [dict(score=d['score'], name=d['label']) for  d in tmp_dict.get_list('enrichments.topics')]
    new_dict['brand_safety'] = tmp_dict.get_dict('enrichments.brand_safety')
    new_dict["file_name"] = fname

    return benedict(new_dict).flatten()


def reduce_article_topics_external(fname, orig_dict):
    def prep(d, nd):
        dnd = {**nd, **d}
        dnd.pop("wikiLink", None)
        return dnd

    tmp_dict = orig_dict["enriched_article"]
    new_dict = {}
    new_dict['article_id'] = orig_dict['article_id']
    new_dict["file_name"] = fname
    return [prep(d, new_dict) for d in tmp_dict.get_list('enrichments.topics')]


def reduce_article_topics_internal(fname, orig_dict):
    def prep(d, nd):
        dnd = {**nd, **d.pop("media_topic"), "score": d.pop("score")}
        return dnd

    tmp_dict = orig_dict["enriched_article"]
    new_dict = {}
    new_dict['article_id'] = orig_dict['article_id']
    new_dict["file_name"] = fname
    return [prep(d, new_dict) for d in tmp_dict.get_list('enrichments.semantics.media_topic_inquiry_results')]


def reduce_article_named_entities(fname, orig_dict):
    def prep(d, nd):
        tmp = {}
        all_mentions = [f"{mention.get('begin')}-{mention.get('end')}" for mention in d.get("mentions", [])]
        tmp["mentions"] = ",".join(all_mentions)
        named_entity = d.get('named_entity', {})
        tmp["name"] = named_entity.get('name')
        tmp["type"] = named_entity.get('type')
        tmp["score"] = d.get('score')
        tmp["saliency"] = d.get('saliency')
        dnd = {**tmp, **nd}
        return dnd

    tmp_dict = orig_dict["enriched_article"]
    new_dict = {}
    new_dict['article_id'] = orig_dict['article_id']
    new_dict["file_name"] = fname
    return [prep(d, new_dict) for d in tmp_dict.get_list('enrichments.semantics.named_entity_inquiry_results')]


def reduce_article_categories_external(fname, orig_dict):
    def prep(d, nd):
        dnd = {**nd, **d}
        dnd.pop("classifierId", None)
        return dnd

    tmp_dict = orig_dict["enriched_article"]
    new_dict = {}
    new_dict['article_id'] = orig_dict['article_id']
    new_dict["file_name"] = fname
    return [prep(d, new_dict) for d in tmp_dict.get_list('enrichments.categories')]


def reduce_article_userneeds_fixed_set(fname, orig_dict):
    tmp_dict = orig_dict["enriched_article"]
    new_dict = {}
    new_dict['article_id'] = orig_dict['article_id']
    new_dict['userneed'] = tmp_dict.get_dict('enrichments.userneeds.scores')
    new_dict["file_name"] = fname
    return benedict(new_dict).flatten()


def reduce_article_topics_fixed_set(fname, orig_dict):
    tmp_dict = orig_dict["enriched_article"]
    new_dict = {}
    new_dict['article_id'] = orig_dict['article_id']
    new_dict['topic'] = tmp_dict.get_dict('enrichments.ci_topics_v2')
    new_dict["file_name"] = fname
    return benedict(new_dict).flatten()


def reduce_article_sensitive_content_fixed_set(fname, orig_dict):
    tmp_dict = orig_dict["enriched_article"]
    new_dict = {}
    new_dict['article_id'] = orig_dict['article_id']
    new_dict['sensitive_topic'] = tmp_dict.get_dict('enrichments.garm.categories')
    new_dict["file_name"] = fname
    return benedict(new_dict).flatten()


print("Starting preflight...")
for fn in tqdm.tqdm(file_list[:1], desc="File"):
    file_name = mapping_files[str(fn)]
    gzipfile = s3_client.get_object(Bucket=BUCKET_NAME, Key=str(fn))["Body"]
    content = TextIOWrapper(gzipfile)
    for cnt, l in tqdm.tqdm(enumerate(content), desc="Lines"):
        tmp_dict = benedict(l)
        raw_collector.append(tmp_dict.copy())
        collector_article.append(reduce_article(file_name, tmp_dict))

        collector_internal_topics.extend(reduce_article_topics_internal(file_name, tmp_dict))
        collector_external_topics.extend(reduce_article_topics_external(file_name, tmp_dict))
        collector_external_categories.extend(reduce_article_categories_external(file_name, tmp_dict))
        collector_external_entities.extend(reduce_article_named_entities(file_name, tmp_dict))
        collector_fixed_set_userneeds.append(reduce_article_userneeds_fixed_set(file_name, tmp_dict))
        collector_fixed_set_topics.append(reduce_article_topics_fixed_set(file_name, tmp_dict))
        collector_fixed_set_sensitive.append(reduce_article_sensitive_content_fixed_set(file_name, tmp_dict))

        if cnt > limit:
            break

df_article = pd.DataFrame(collector_article)
df_internal_topics = pd.DataFrame(collector_internal_topics)
df_external_topics = pd.DataFrame(collector_external_topics)
df_external_categories = pd.DataFrame(collector_external_categories)
df_external_entities = pd.DataFrame(collector_external_entities)
df_fixed_set_sensitive = pd.DataFrame(collector_fixed_set_sensitive)
df_fixed_set_topics = pd.DataFrame(collector_fixed_set_topics)
df_fixed_set_userneeds = pd.DataFrame(collector_fixed_set_userneeds)
df_raw = pd.DataFrame(raw_collector)

# %%
workbook = {
    "df_article": df_article.to_dict('records'),
    "df_internal_topics":df_internal_topics.to_dict('records'),
    "df_external_topics":df_external_topics.to_dict('records'),
    "df_external_categories":df_external_categories.to_dict('records'),
    "df_external_entities":df_external_entities.to_dict('records'),
    "df_fixed_set_sensitive":df_fixed_set_sensitive.to_dict('records'),
    "df_fixed_set_topics":df_fixed_set_topics.to_dict('records'),
    "df_fixed_set_userneeds":df_fixed_set_userneeds.to_dict('records'),
}

# %%
for k, v in workbook.items():
    p.save_as(records=v, dest_file_name=f"data_dpg_testdata/preflight/reduced_articles_{k}.csv")

# df_article.to_csv("data_dpg_testdata/preflight/reduced_articles.csv", index=None)
df_raw.to_csv("data_dpg_testdata/preflight/raw_articles.csv", index=None)

print("Ending preflight...")
# %%
# Full run
limit = None
update_freq = 10000
print("Starting full run...")

def create_path(name):
    return f"./data_dpg_testdata/reduced/reduced_articles_{name}.csv"

file_df_article = csv.DictWriter(io.open(create_path('df_article'), "w"), fieldnames=df_article.columns.to_list())
file_df_internal_topics = csv .DictWriter(io.open(create_path('df_internal_topics'), "w"), fieldnames=df_internal_topics.columns.to_list())
file_df_external_topics = csv .DictWriter(io.open(create_path('df_external_topics'), "w"), fieldnames=df_external_topics.columns.to_list())
file_df_external_categories = csv .DictWriter(io.open(create_path('df_external_categories'), "w"), fieldnames=df_external_categories.columns.to_list())
file_df_external_entities = csv .DictWriter(io.open(create_path('df_external_entities'), "w"), fieldnames=df_external_entities.columns.to_list())
file_df_fixed_set_sensitive = csv.DictWriter(io.open(create_path('df_fixed_set_sensitive'), "w"), fieldnames=df_fixed_set_sensitive.columns.to_list())
file_df_fixed_set_topics = csv.DictWriter(io.open(create_path('df_fixed_set_topics'), "w"), fieldnames=df_fixed_set_topics.columns.to_list())
file_df_fixed_set_userneeds = csv.DictWriter(io.open(create_path('df_fixed_set_userneeds'), "w"), fieldnames=df_fixed_set_userneeds.columns.to_list())

file_df_article.writeheader()
file_df_internal_topics.writeheader()
file_df_external_topics.writeheader()
file_df_external_categories.writeheader()
file_df_fixed_set_sensitive.writeheader()
file_df_fixed_set_topics.writeheader()
file_df_fixed_set_userneeds.writeheader()
for fn in tqdm.tqdm(file_list, desc="File"):
    file_name = mapping_files[str(fn)]
    gzipfile = s3_client.get_object(Bucket=BUCKET_NAME, Key=str(fn))["Body"]
    content = TextIOWrapper(gzipfile)
    for cnt, l in tqdm.tqdm(enumerate(content), desc="Lines"):
        tmp_dict = benedict(l)

        file_df_article.writerow(reduce_article(file_name, tmp_dict))
        file_df_internal_topics.writerows(reduce_article_topics_internal(file_name, tmp_dict))
        file_df_external_topics.writerows(reduce_article_topics_external(file_name, tmp_dict))
        file_df_external_categories.writerows(reduce_article_categories_external(file_name, tmp_dict))
        file_df_external_entities.writerows(reduce_article_named_entities(file_name, tmp_dict))
        file_df_fixed_set_userneeds.writerow(reduce_article_userneeds_fixed_set(file_name, tmp_dict))
        file_df_fixed_set_topics.writerow(reduce_article_topics_fixed_set(file_name, tmp_dict))
        file_df_fixed_set_sensitive.writerow(reduce_article_sensitive_content_fixed_set(file_name, tmp_dict))
        
print("Ending full run...")
# %%
