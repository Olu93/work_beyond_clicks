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

# FILENAME is not really needed. But remains for example purposes.
BUCKET_NAME = "recosearch-nps-bucket-shared-with-utrecht-university"
FILE_NAME = "20220909-test-on-tracking-event-file-partition-and-compression/tracking_events_single_file_compressed_json_2022-06-30/part-00000-d667af8c-fe5d-4c75-b144-400a5560425d-c000.json.gz"
time_filter = "20221114"

# list all directories and files inside the bucket
s3_objects = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
file_list = [pathlib.Path(item["Key"]) for item in s3_objects["Contents"] if "json.gz" in item["Key"] and (time_filter in item["Key"])]
print(file_list)
# %%
mapping_files = {str(f):i for i, f in enumerate(file_list)}
mapping_files
json.dump(mapping_files,io.open("./data_dpg_testdata/reduced/mapping_view_files.json", "w"))

# %%
counter = collections.defaultdict(int)
raw_collector = collections.deque()
view_collector = collections.deque()
interaction_collector = collections.deque()
mapping_geo = collections.defaultdict(str)
mapping_city = collections.defaultdict(lambda: len(mapping_city)+1)
mapping_refferer = collections.defaultdict(lambda: len(mapping_refferer)+1)
limit = 10000
update_freq = 1000

# Helper function for the reduction part.
def reduce_event(geo_mapping, city_mapping, refferer_mapping, tmp_dict):
    # As a user can only have either an android id, or an apple id or a privacy wall id, we combine these columns. 
    tmp_dict["QUASI_USER_ID"] = tmp_dict.pop("ANDROID_ID", None) or tmp_dict.pop("APPLE_ID", None) or tmp_dict.pop("PRIVACYWALL_ID", None) or ""
    # If a user is logged in, the user has a GIGYA-ID. We only save whther the user is logged in or not. 
    tmp_dict["IS_LOGGED_IN"] = (tmp_dict.pop("GIGYA_ID", None) is not None)*1
    
    # Remove "article__" prefix from some article ids
    tmp_dict["ARTICLE_ID"] = tmp_dict["ARTICLE_ID"].replace("article__", "")
    
    # Creates two look up tables.
    # Look up for mapping from geo-region-abbreviation to region name and timezone 
    geo_mapping[tmp_dict.get("GEO_REGION", None)] = {"region_name":tmp_dict.pop("GEO_REGION_NAME", None), "timezone": tmp_dict.pop("GEO_TIMEZONE", None)} 
    # Look up for mapping from city to id 
    city_mapping[tmp_dict.get("GEO_CITY", None)]    
    tmp_dict["GEO_CITY"] = city_mapping[tmp_dict.get("GEO_CITY", None)]
    # Look up for mapping from city to id 
    refferer_mapping[tmp_dict.get("REFR_URLHOST", None)]    
    tmp_dict["REFR_URLHOST"] = refferer_mapping[tmp_dict.get("REFR_URLHOST", None)]
    
    # Removal of unneeded information in the data
    tmp_dict.pop("DVCE_CREATED_TSTAMP", None) 
    # DOMAIN USER- and SESSIONID are created by the snow plow system but DPG uses PRIVACYWALL_ID instead.        
    tmp_dict.pop("DOMAIN_USERID", None)        
    tmp_dict.pop("DOMAIN_SESSIONID", None)
    
    tmp_dict.pop("PLATFORM", None)        
    # Removal-Reason: Information is already available in the article data.
    tmp_dict.pop("PAGE_TITLE", None)        
    # Removal-Reason: Information is already available in the article data.
    tmp_dict.pop("PAGE_URLHOST", None)        
    # Removal-Reason: Information is already available in the article data.
    tmp_dict.pop("PAGE_URLPATH", None)        
    # Removal-Reason: Type is always 'article' 
    tmp_dict.pop("PAGE_TYPE", None)        
    # Removal-Reason: Holds redundant information to field 'REFR_URLHOST'
    tmp_dict.pop("PAGE_REFERRER", None)        
    # Removal-Reason: Holds redundant information to field 'REFR_URLHOST'
    tmp_dict.pop("REFR_SOURCE", None) 
    
    # Seperates all combined privacy configurations into binary values.        
    if "PRIVACY_SETTINGS" in tmp_dict:
        psettings = tmp_dict["PRIVACY_SETTINGS"]
        tmp_dict["privacy_functional"] = ("functional" in psettings)*1
        tmp_dict["privacy_analytics"] = ("analytics" in psettings)*1
        tmp_dict["privacy_target_advertising"] = ("target_advertising" in psettings)*1
        tmp_dict["privacy_personalisation"] = ("personalisation" in psettings)*1
        tmp_dict["privacy_non-personalised_ads"] = ("non-personalised_ads" in psettings)*1
        tmp_dict["privacy_marketing"] = ("marketing" in psettings)*1
        tmp_dict["privacy_social_media"] = ("social_media" in psettings)*1
        tmp_dict["privacy_geo_location"] = ("geo_location" in psettings)*1
        tmp_dict["privacy_advertising"] = ("advertising_" in psettings)*1
        del tmp_dict["PRIVACY_SETTINGS"]
    return tmp_dict


# This is a pre-flight to get all necessary columns beforehand.
limit = 10000
update_freq = 1000
for file_name in file_list:
    pbar = tqdm.tqdm(range(limit), total=limit)
    with gzip.GzipFile(fileobj=s3_client.get_object(Bucket=BUCKET_NAME, Key=str(file_name))["Body"]) as gzipfile:
        content = TextIOWrapper(gzipfile)
        for cnt, l in enumerate(content):
            tmp_dict = json.loads(l)
            raw_collector.append(tmp_dict.copy())
            ev_name = tmp_dict.pop('EVENT_NAME', 'no-event')
            tmp_dict = reduce_event(mapping_geo, mapping_city, mapping_refferer, tmp_dict)
            tmp_dict["file_name"] = mapping_files[str(file_name)]            
            if ev_name in ["screen_view", "page_view"]:
                view_collector.append(tmp_dict)
                # file_reduced_views.writerow(tmp_dict)
            else:
                interaction_collector.append(tmp_dict)            
            counter[ev_name] += 1
            if ((cnt+1)%update_freq) == 0:
                pbar.update(update_freq)
            if cnt > limit:
                break

    pbar.close()        

df_views = pd.DataFrame(view_collector)
df_interactions = pd.DataFrame(interaction_collector)
df_raw = pd.DataFrame(raw_collector)
df_views.to_csv("data_dpg_testdata/preflight/reduced_views.csv", index=None)
df_interactions.to_csv("data_dpg_testdata/preflight/reduced_interactions.csv", index=None)
df_raw.to_csv("data_dpg_testdata/preflight/raw_views.csv", index=None)
data = bytes(df_views.to_csv(index=None), encoding="utf-8")
s_out = gzip.compress(data)
io.open("data_dpg_testdata/preflight/compressed_views.csv.gz", mode="wb").write(s_out)

# %%
# Full run
limit = None
update_freq = 100000
with io.open(f"./data_dpg_testdata/reduced/reduced_views.csv", "w") as file_reduced_views:
    with io.open(f"./data_dpg_testdata/reduced/reduced_interactions.csv", "w") as file_reduced_interactions:
        writer_reduced_views = csv.DictWriter(file_reduced_views, fieldnames=list(df_views.columns))
        writer_reduced_interactions = csv.DictWriter(file_reduced_interactions, fieldnames=list(df_interactions.columns))
        writer_reduced_views.writeheader()
        writer_reduced_interactions.writeheader()
        for file_name in file_list:
            pbar = tqdm.tqdm(range(limit) if limit else None, total=limit)
            with gzip.GzipFile(fileobj=s3_client.get_object(Bucket=BUCKET_NAME, Key=str(file_name))["Body"]) as gzipfile:
                content = TextIOWrapper(gzipfile)
                for cnt, l in enumerate(content):
                    tmp_dict = json.loads(l)
                    ev_name = tmp_dict.pop('EVENT_NAME', 'no-event')
                    tmp_dict = reduce_event(mapping_geo, mapping_city, mapping_refferer, tmp_dict)
                    tmp_dict["file_name"] = mapping_files[str(file_name)]
                    if ev_name in ["screen_view", "page_view"]:
                        writer_reduced_views.writerow(tmp_dict)
                    else:
                        writer_reduced_interactions.writerow(tmp_dict)
                    counter[ev_name] += 1
                    if ((cnt+1)%update_freq) == 0:
                        pbar.update(update_freq)
                        json.dump(mapping_geo, io.open("data_dpg_testdata/reduced/mapping_geo.json", "w"))
                        json.dump(mapping_city, io.open("data_dpg_testdata/reduced/mapping_city.json", "w"))
                        json.dump(mapping_refferer, io.open("data_dpg_testdata/reduced/mapping_refferer.json", "w"))
    pbar.close()        
counter
# %%
# Save all the mappings

# %%
