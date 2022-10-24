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
file_list = [pathlib.Path(item["Key"]) for item in s3_objects["Contents"] if "json.gz" in item["Key"] and "single_file" in item["Key"]]
print(file_list)

# %%
# read a file in json lines format and print its first record
# Note: the "Body" field contains the data as a botocore.response.StreamingBody object,
# and .read() will read all the content of the object
# json_list = s3_object["Body"].read().decode("utf-8").strip().split("\n")
# print(json.loads(json_list[0]))
# %%
# try to delete the bucket - you should get an access denied error because you have read-only permissions
# s3_client.delete_bucket(Bucket=bucket_name)

# %%

# cnt = 0
# s3_object = s3_client.get_object(Bucket=bucket_name, Key=FILE_NAME)
# print("Loading File")
# s3_file_content = s3_object['Body'].read()
# print("Loading Complete")
# with gzip.GzipFile(None, "rb", fileobject=BytesIO(s3_file_content)) as file:
#     print("Opening Zip-File")
#     for cnt, l in enumerate(file):
#         print(l)
#         if cnt > 10:
#             break


# %%
class ProgressPercentage(object):
    ''' Progress Class
    Class for calculating and displaying download progress
    '''
    def __init__(self, client, bucket, filename):
        ''' Initialize
        initialize with: file name, file size and lock.
        Set seen_so_far to 0. Set progress bar length
        '''
        self._filename = filename
        self._size = client.head_object(Bucket=bucket, Key=filename)['ContentLength']
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self.prog_bar_len = 80

    def __call__(self, bytes_amount):
        ''' Call
        When called, increments seen_so_far by bytes_amount,
        calculates percentage of seen_so_far/total file size 
        and prints progress bar.
        '''
        # To simplify we'll assume this is hooked up to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            ratio = round((float(self._seen_so_far) / float(self._size)) * (self.prog_bar_len - 6), 1)
            current_length = int(round(ratio))

            percentage = round(100 * ratio / (self.prog_bar_len - 6), 1)

            bars = '+' * current_length
            output = bars + ' ' * (self.prog_bar_len - current_length - len(str(percentage)) - 1) + str(percentage) + '%'

            if self._seen_so_far != self._size:
                sys.stdout.write(output + '\r')
            else:
                sys.stdout.write(output + '\n')
            sys.stdout.flush()


# %%
# s3 = boto3.resource('s3', aws_access_key_id=os.environ["ACCESS_KEY"], aws_secret_access_key=os.environ["SECRET_KEY"])
# binary_file = BytesIO()
# progress = ProgressPercentage(s3_client, BUCKET_NAME, FILE_NAME)
# s3.Bucket(BUCKET_NAME).download_fileobj(FILE_NAME, binary_file, Callback=progress)
# gzip_file = gzip.GzipFile(None, 'rb', fileobj=binary_file)
# decompressed = gzip_file.decompress()
# decompressed
# %%
import csv
counter = collections.defaultdict(int)
raw_collector = collections.deque()
view_collector = collections.deque()
interaction_collector = collections.deque()
geo_mapping = collections.defaultdict(str)
city_mapping = collections.defaultdict(lambda: len(city_mapping)+1)
limit = 10000
update_freq = 1000

def reduce_event(geo_mapping, city_mapping, tmp_dict):
    tmp_dict["QUASI_USER_ID"] = tmp_dict.pop("ANDROID_ID", None) or tmp_dict.pop("APPLE_ID", None) or tmp_dict.pop("PRIVACYWALL_ID", None) or ""
    tmp_dict["IS_LOGGED_IN"] = (tmp_dict.pop("GIGYA_ID", None) is not None)*1
    geo_mapping[tmp_dict.pop("GEO_REGION", None)] = {"region_name":tmp_dict.pop("GEO_REGION_NAME", None), "timezone": tmp_dict.pop("GEO_TIMEZONE", None)} 
    city_mapping[tmp_dict.get("GEO_CITY", None)]    
    tmp_dict["GEO_CITY"] = city_mapping[tmp_dict.get("GEO_CITY", None)]
            
    if "DVCE_CREATED_TSTAMP" in tmp_dict:
        del tmp_dict["DVCE_CREATED_TSTAMP"]
    if "DOMAIN_USERID" in tmp_dict:
        del tmp_dict["DOMAIN_USERID"]
    if "DOMAIN_SESSIONID" in tmp_dict:
        del tmp_dict["DOMAIN_SESSIONID"]
    if "REFR_URLHOST" in tmp_dict:
        del tmp_dict["REFR_URLHOST"]
    if "PLATFORM" in tmp_dict:
        del tmp_dict["PLATFORM"]
    if "PAGE_TITLE" in tmp_dict:
        del tmp_dict["PAGE_TITLE"]
    if "PAGE_TYPE" in tmp_dict:
        del tmp_dict["PAGE_TYPE"]
    if "PAGE_REFERRER" in tmp_dict:
        del tmp_dict["PAGE_REFERRER"]
    if "PAGE_URLHOST" in tmp_dict:
        del tmp_dict["PAGE_URLHOST"]
    if "PAGE_URLPATH" in tmp_dict:
        del tmp_dict["PAGE_URLPATH"]
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

def remove_interaction_cols(tmp_dict):
    if "SE_VALUE" in tmp_dict:
        del tmp_dict["SE_VALUE"]
    if "SE_CATEGORY" in tmp_dict:
        del tmp_dict["SE_CATEGORY"]
    if "SE_LABEL" in tmp_dict:
        del tmp_dict["SE_LABEL"]
    if "SE_ACTION" in tmp_dict:
        del tmp_dict["SE_ACTION"]

for file_name in file_list:
    pbar = tqdm.tqdm(range(limit), total=limit)
    with gzip.GzipFile(fileobj=s3_client.get_object(Bucket=BUCKET_NAME, Key=str(file_name))["Body"]) as gzipfile:
        content = TextIOWrapper(gzipfile)
        for cnt, l in enumerate(content):
            tmp_dict = json.loads(l)
            raw_collector.append(tmp_dict.copy())
            ev_name = tmp_dict.pop('EVENT_NAME', 'no-event')
            tmp_dict = reduce_event(geo_mapping, city_mapping, tmp_dict)
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
df_views.to_csv("data_dpg_testdata/reduced_views2.csv", index=None)
df_interactions.to_csv("data_dpg_testdata/reduced_interactions2.csv", index=None)
df_raw.to_csv("data_dpg_testdata/raw2.csv", index=None)
# %%
limit = None
update_freq = 10000
for file_name in file_list:
    pbar = tqdm.tqdm(range(limit) if limit else None, total=limit)
    with gzip.GzipFile(fileobj=s3_client.get_object(Bucket=BUCKET_NAME, Key=str(file_name))["Body"]) as gzipfile:
        with io.open(f"./data_dpg_testdata/reduced_views-{file_name.stem}.csv", "w") as file_reduced_views:
            with io.open(f"./data_dpg_testdata/reduced_interactions-{file_name.stem}.csv", "w") as file_reduced_interactions:
                writer_reduced_views = csv.DictWriter(file_reduced_views, fieldnames=list(df_views.columns))
                writer_reduced_interactions = csv.DictWriter(file_reduced_interactions, fieldnames=list(df_interactions.columns))
                writer_reduced_views.writeheader()
                writer_reduced_interactions.writeheader()
                content = TextIOWrapper(gzipfile)
                for cnt, l in enumerate(content):
                    tmp_dict = json.loads(l)
                    # raw_collector.append(tmp_dict.copy())
                    ev_name = tmp_dict.pop('EVENT_NAME', 'no-event')
                    tmp_dict = reduce_event(geo_mapping, city_mapping, tmp_dict)
                    if ev_name in ["screen_view", "page_view"]:
                        # view_collector.append(tmp_dict)
                        writer_reduced_views.writerow(tmp_dict)
                    else:
                        # interaction_collector.append(tmp_dict)            
                        writer_reduced_interactions.writerow(tmp_dict)
                    counter[ev_name] += 1
                    if ((cnt+1)%update_freq) == 0:
                        pbar.update(update_freq)
                    # if cnt > limit:
                    #     break

    pbar.close()        
counter
# %%

# df_views.to_csv("data_dpg_testdata/reduced_views.csv", index=None)
# df_interactions.to_csv("data_dpg_testdata/reduced_interactions.csv", index=None)
# df_raw.to_csv("data_dpg_testdata/raw.csv", index=None)
json.dump(geo_mapping, io.open("data_dpg_testdata/mapping_geo.json", "w"))
json.dump(city_mapping, io.open("data_dpg_testdata/mapping_city.json", "w"))
# %%
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
