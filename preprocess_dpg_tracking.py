# %%
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
from io import BytesIO, TextIOWrapper
import gzip
import collections
import tqdm
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
print([item["Key"] for item in s3_objects["Contents"]])

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
counter = collections.defaultdict(int)
collector = collections.deque()
limit = 10000
update_freq = 500
pbar = tqdm.tqdm(range(limit), total=limit)
with gzip.GzipFile(fileobj=s3_client.get_object(Bucket=BUCKET_NAME, Key=FILE_NAME)["Body"]) as gzipfile:
    content = TextIOWrapper(gzipfile)
    for cnt, l in enumerate(content):
        
        tmp_dict = json.loads(l)
        ev_name = tmp_dict.get('EVENT_NAME')
        # tmp_dict["QUASI_USER_ID"] = tmp_dict.get("ANDROID_ID") or tmp_dict.get("APPLE_ID") or tmp_dict.get("PRIVACYWALL_ID") or ""
        # tmp_dict["QUASI_USER_ID"] = tmp_dict.get("APP_ID", "no-app") + "-" + tmp_dict.get("QUASI_USER_ID", "") 
        # if "GEO_REGION" in tmp_dict:
        #     del tmp_dict["GEO_REGION"]
        # if "GEO_REGION_NAME" in tmp_dict:
        #     del tmp_dict["GEO_REGION_NAME"]
        # if "GEO_ZIPCODE" in tmp_dict:
        #     del tmp_dict["GEO_ZIPCODE"]
        # # del tmp_dict["PLATFORM"]
        # # del tmp_dict["PRIVACYWALL_ID"]
        # # del tmp_dict["PRIVACY_SETTINGS"]
        # if "REFR_URLHOST" in tmp_dict:
        #     del tmp_dict["REFR_URLHOST"]
        # if "PAGE_REFERRER" in tmp_dict:
        #     del tmp_dict["PAGE_REFERRER"]
        # if "PAGE_URLHOST" in tmp_dict:
        #     del tmp_dict["PAGE_URLHOST"]
        # if "SE_VALUE" in tmp_dict:
        #     del tmp_dict["SE_VALUE"]
        # if "SE_CATEGORY" in tmp_dict:
        #     del tmp_dict["SE_CATEGORY"]
        # Ask about SE ACTIONS
        collector.append(tmp_dict)
        counter[ev_name] += 1
        if ((cnt+1)%update_freq) == 0:
            pbar.update(update_freq)
            # pbar.flush()
            # sys.stdout.flush()
        if cnt > limit:
            break
pbar.close()        
counter
# %%
df = pd.DataFrame(collector)
df.head(5)
# %%


# %%
for col in df.columns:
    print(f"---------- Column {col} -----------")
    display(df[col].value_counts())

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
