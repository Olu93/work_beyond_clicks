# %%
import gzip
import csv
import json
import pandas as pd

# df = pd.read_csv('tracking_events_2022-06-26.csv.gz', compression='gzip', sep=',', quotechar='"', error_bad_lines=False)
# %%
# df.head(100)
# %%
import gzip
data_array = []
with gzip.open('tracking_events_2022-06-26.csv.gz', 'r') as f:
    for idx, line in enumerate(f):
        data_array.append(str(line).strip().split(","))
        if idx > 100:
            break

df = pd.DataFrame(data_array[1:])
df        
# %%
["b'DERIVED_TSTAMP",
 'DVCE_CREATED_TSTAMP',
 'APP_ID', # Is not unique - Needs brand
 'EVENT_NAME',
 'PAGE_TYPE',
 'ARTICLE_ID',
 'PRIVACY_SETTINGS',
 'GEO_CITY',
 'GEO_COUNTRY',
 'GEO_REGION',
 'GEO_REGION_NAME',
 'GEO_TIMEZONE',
 'GEO_ZIPCODE',
 'PAGE_REFERRER',
 'PAGE_TITLE',
 'PAGE_URLHOST',
 'PAGE_URLPATH',
 'PLATFORM',
 'REFR_MEDIUM',
 'REFR_SOURCE',
 'REFR_URLHOST',
 'GIGYA_ID',
 'PRIVACYWALL_ID',
 'ANDROID_ID',
 'APPLE_ID',
 'DOMAIN_SESSIONID',
 "DOMAIN_USERID\\n'"]
# Snowplow inspector
# pwid is id if you are not logged in - first party cookie
# privacySettings includes personalisation tag
# pwid remains for months
# On mobile pwid is not available but android idx
# Gigya id are logged in Users
# first_publication_ts is actually updated