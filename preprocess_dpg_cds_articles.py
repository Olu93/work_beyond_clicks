# %%
from typing import List
import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import flatdict
import numpy as np
# %%
df = pd.read_json('data_dpg_testdata/raw_cds_data_json_2022-04-01_2022-07-04.jsonl', orient='records', lines=True)
df


# %%
def transform_level_1(x: pd.DataFrame):
    d = flatdict.FlatDict(x, delimiter='.')
    return d


def transform_1_to_1_list(x: pd.DataFrame):
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
sub_images.columns = [f"images.{el}" for el in sub_images.columns]
sub_brand_publication = sub_enriched["brand_publications"].apply(transform_1_to_1_list).apply(pd.Series)
sub_brand_publication.columns = [f"enriched_article.{el}" for el in sub_brand_publication.columns]
sub_brand_categories = sub_enriched["categories"].apply(lambda x: "|".join(x))
sub_brand_categories.name = f"enriched_article.{sub_brand_categories.name}"
sub_brand_source_keywords = sub_enriched["source_keywords"].apply(lambda x: "|".join(x))
sub_brand_source_keywords.name = f"enriched_article.{sub_brand_source_keywords.name}"
sub_complex_enrichments = sub_enriched[[col for col in sub_enriched.select_dtypes('object').columns
                                        if 'enrichment' in col]].drop(["enrichments.userneeds.max", "enrichments.language"], axis=1)
sub_complex_enrichments

# %%
sub_top_level = df.drop('enriched_article', axis=1)
sub_remaining = sub_enriched.drop([
    "cds_content_id",
    "images",
    "brand_publications",
    "brand",
    "categories",
    "source_keywords",
], axis=1).drop(sub_complex_enrichments.columns, axis=1)
sub_remaining

# %%
def transform_extract_media_topic(x: list):
    info_tuples = [list((el['media_topic']['id'], el['media_topic']['name'], el['score'], el['enabled']) for el in l) for l in x]
    list_bundles = [list(zip(*l)) for l in info_tuples]
    return list_bundles


cols = ["id", "name", "score", "enabled"]
sub_media_topic_enrichments = pd.DataFrame(
    sub_complex_enrichments["enrichments.semantics.media_topic_inquiry_results"].to_frame().apply(transform_extract_media_topic).iloc[:, 0].to_list(),
    columns=[f"media_topic.{c}" for c in cols])
sub_media_topic_enrichments


# %%
def sub_extract_named_entity(el):
    named_entity = el.get('named_entity', {})
    name = named_entity.get('name')
    _type = named_entity.get('type')
    wiki_link = named_entity.get('wiki_link', {})
    wiki_id = wiki_link.get('id')
    wiki_title = wiki_link.get('title')
    wiki_url = wiki_link.get('url')
    score = el.get('score')
    confidence = el.get('confidence')
    keyword = el.get('keyword')
    saliency = el.get('saliency')
    mentions = ",".join([f"{m['begin']}-{m['end']}" for m in el.get('mentions', [])])

    return (
        name,
        _type,
        score,
        confidence,
        keyword,
        saliency,
        mentions,
        wiki_id,
        wiki_title,
        wiki_url,
    )


def transform_extract_named_entity(x: list):
    info_tuples = [list(sub_extract_named_entity(el) for el in l) for l in x]
    list_bundles = [list(zip(*l)) for l in info_tuples]
    return list_bundles


cols = [
    "name",
    "type",
    "score",
    "confidence",
    "keyword",
    "saliency",
    "mentions",
    "wiki_id",
    "wiki_title",
    "wiki_url",
]
sub_named_entity_enrichments = pd.DataFrame(
    sub_complex_enrichments["enrichments.semantics.named_entity_inquiry_results"].to_frame().apply(transform_extract_named_entity).iloc[:, 0].to_list(),
    columns=[f"named_entity.{c}" for c in cols])
sub_named_entity_enrichments


# %%
def sub_extract_topics(el):
    name = el.get('label')
    id = el.get('id')
    score = el.get('score')
    wiki_url = el.get('wikiLink')
    wiki_data_id = el.get('wikidataId')

    return (
        name,
        score,
        id,
        wiki_url,
        wiki_data_id,
    )


def transform_extract_topics(x: list):
    info_tuples = [list(sub_extract_topics(el) for el in l) for l in x]
    list_bundles = [list(zip(*l)) for l in info_tuples]
    return list_bundles


cols = [
    "name",
    "score",
    "id",
    "wiki_url",
    "wiki_data_id",
]
sub_topic_enrichments = pd.DataFrame(sub_complex_enrichments["enrichments.topics"].to_frame().apply(transform_extract_topics).iloc[:, 0].to_list(),
                                     columns=[f"topics.{c}" for c in cols])
sub_topic_enrichments


# %%
def sub_extract_categories(el):
    name = el.get('label')
    score = el.get('score')
    id = el.get('categoryId')
    classifier_id = el.get('classifierId')

    return (
        name,
        score,
        id,
        classifier_id,
    )


def transform_extract_categories(x: list):
    info_tuples = [list(sub_extract_categories(el) for el in l) for l in x]
    list_bundles = [list(zip(*l)) for l in info_tuples]
    return list_bundles


cols = [
    "name",
    "score",
    "id",
    "classifier_id",
]
sub_category_enrichments = pd.DataFrame(sub_complex_enrichments["enrichments.categories"].to_frame().apply(transform_extract_categories).iloc[:, 0].to_list(),
                                        columns=[f"categories.{c}" for c in cols])
sub_category_enrichments


# %%
def sub_extract_imgperson(el):
    age = el.get('age')
    race = el.get('race')
    gender = el.get('gender')

    return (
        age,
        race,
        gender,
    )


def transform_extract_imgperson(x: list):
    # tmp = [isinstance(l, float) for l in x]
    info_tuples = [list(sub_extract_imgperson(el) for el in l) for l in x]
    list_bundles = [list(zip(*l)) for l in info_tuples]
    return list_bundles


cols = [
    "age",
    "race",
    "gender",
]

tmp_df = sub_complex_enrichments["enrichments.image_persons"].copy()
tmp_df_nan_indices = ~tmp_df.map(lambda x: isinstance(x, list))
tmp_df.loc[tmp_df_nan_indices] = [[] for _ in range(np.sum(tmp_df_nan_indices))]  # [np.empty(0,dtype=float)]*np.sum(tmp_df_nan_indices)
sub_imgperson_enrichments = pd.DataFrame(tmp_df.to_frame().apply(transform_extract_imgperson).iloc[:, 0].to_list(), columns=[f"imgperson.{c}" for c in cols])
sub_imgperson_enrichments
# %%
# %%
complex_dict_cols = [
    sub_images,
    sub_brand_publication,
    sub_brand_categories,
    sub_brand_source_keywords,
]

complex_list_cols = [
    sub_media_topic_enrichments,
    sub_named_entity_enrichments,
    sub_topic_enrichments,
    sub_category_enrichments,
    sub_imgperson_enrichments,
]

df_all_merged = sub_top_level.join(sub_remaining).join(complex_dict_cols).join(complex_list_cols)
df_all_merged
# %%
df_all_merged[:15].T.to_csv("figs/cds_column_overview.csv")
# %%
