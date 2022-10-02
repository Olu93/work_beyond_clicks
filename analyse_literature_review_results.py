# %%
from turtle import width
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import itertools as it
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets
from fastcluster import linkage
# def set_style():
sns.set_context("paper")
plt.style.use(['seaborn-white', 'seaborn-paper'])
# matplotlib.rc("font", family="Times New Roman")

#%%
C_TYPE_PAPER = "Type Of Paper"
C_PLATFORM = "Platform"
C_DIVERSITY_TYPE = 'Diversity-Type'
C_EVALUATION_TYPE = 'Additional Details'
C_YEAR = "Year"
C_TITLE = "Title"
C_VALUE_GROUP = "Value Group"
C_METRICS_GROUP = "Metric Group"
C_VALUE = "Value"
C_MENTIONS = "Mentions"
C_GRP_UX = "UX Values"
C_GRP_NEWS = "News Values"
C_GRP_RESPONSIBILITY = "Responsibility Values"
C_GRP_GOTO = "Standard Values"
C_GRP_TECHNICAL = "Technical Values"
C_GRP_OVERALL = "Overall"
C_COOC = "Co-Occurrence"
C_COMBINATION = "Combination"
C_COMPLEMENT = "Complement"
C_ROLLING = "Rolling"
C_METRICS_ACC = "Metric: Accuracy-Family"
C_METRICS_CTR = "Metric: CTR & nDCG"
C_METRICS_OTHER = "Metric: Others"
C_METRICS_DIVERSITY = "Metric: ILL"

data = pd.read_csv(io.open('data_literature_review/data_post_screening.csv', 'r', encoding='utf-8'))
data = data.set_index("ID")
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
data = data[[col for col in data.columns if not ("Spalte" in col)]]
data

# %%
data.columns = [col.replace("3", "").strip().strip("_").title() for col in data.columns]
# %%
all_relevant_cols = [c for c in data.columns if ("X" in data[c].unique()) or ("x" in data[c].unique())]
all_relevant_cols
# %%
# %%
data[all_relevant_cols] = data[all_relevant_cols]
data[all_relevant_cols] = data[all_relevant_cols].replace(["X", "x"], 1)
data[all_relevant_cols] = data[all_relevant_cols].replace([""], 0)
data[all_relevant_cols] = data[all_relevant_cols].fillna(0)
data[all_relevant_cols] = data[all_relevant_cols].infer_objects()
data[all_relevant_cols] = data[all_relevant_cols].astype(float)
data = data[data['Not Relevant'] == 0].drop('Not Relevant', axis=1)
data[C_TYPE_PAPER] = data[C_TYPE_PAPER].str.title().replace("Conceptual/Analysis", "Conceptual").replace("Conceptual/Review", "Conceptual")
data[C_YEAR] = pd.to_datetime(data[C_YEAR].astype(str), format="%Y")
data[C_DIVERSITY_TYPE] = data[C_DIVERSITY_TYPE].str.split(',')
data[C_EVALUATION_TYPE] = data[C_EVALUATION_TYPE].str.split(',')
data = data.sort_values(C_YEAR)
data["Recall"] = data["Recall"] + data["Hit-Rate"]

data.dtypes
# %%
# %%

# ['Who', 'Year', 'Author', 'Title', 'DOI', 'Problem Statement',
# 'Problem Statement2', 'Type of Paper3', 'Additional Details3',
# 'Domain3', 'Platform3', 'Diversity3', 'Diversity-Type3', 'Agency3',
# 'Novelty3', 'Context', 'Context type', 'Explainability3',
# 'Democracy3', 'Privacy3', 'Accountability3', 'Trust3',
# 'Transparency3', 'Objectivity3', 'Serendipity3', 'Location3',
# 'Recency3', 'Coverage3', 'Newsworthiness/Relevance', 'Emotion3',
# 'Interests over time (short medium long)', 'Decentralization',
# 'Engagement', 'Temporality of interests', 'Fairness', 'Autonomy 3',
# 'Credibility 3', 'Freedom', 'Censorship2',
# 'instrumentalization/propaganda', 'Fatigue 3', 'Authority',
# 'Popularity', 'Realtime Capabilities', 'Editorial Influence',
# 'Surprise', 'Curiosity', 'Personalisation', 'Saliency', 'Future Impact',
# 'Tradeoff', 'Manipulation Prevention', 'Scalability',
# 'Convenience', 'Utility', 'User Satisfaction',
# 'Journalistic Values', 'Shifting user interests 3', 'Comment',
# 'Comment2', 'Not Relevant', 'Click-Through-Rate3', 'Accuracy3',
# 'Precision3', 'Recall3', 'F1_3', 'Intra-list-diversity3',
# 'Cosine similarity3', 'nDCG3', 'Wilcoxon Signed Ranks Test3', 'AUC',
# 'Duration', 'MRR', 'RMSE', 'Expected Utility', 'WTP',
# 'Other Metrics 3']

COLS_USER_EXPERIENCE = [
    'Shifting User Interests',
    'Engagement',
    'User Satisfaction',
    'Curiosity',
    'Emotion',
    'Serendipity',
    'Fatigue',
    'Surprise',
    'Convenience',
]

COLS_NEWS_VALUES = [
    'Freedom',
    'Authority',
    'Objectivity',
    'Democracy',
    'Journalistic Values',
]

COLS_NEWS_RECOMMENDER_SPECIFIC = [
    'Location',
    'Saliency',
    'Context',
    'Recency',
    'Newsworthiness/Relevance',
]

COLS_RESPONSIBILITY = [
    'Privacy',
    'Explainability',
    'Accountability',
    'Fairness',
    'Trust',
    'Transparency',
    'Manipulation Prevention',
]

COLS_RESPONSIBILITY_AGENCY = [
    'Autonomy',
    'Agency',
    'Future Impact',
]
COLS_GOTO = [
    'Diversity',
    'Popularity',
    'Coverage',
    'Novelty',
]

COLS_TECHNICAL = [
    'Scalability',
    'Realtime Capabilities',
    # 'Personalisation',
    'Utility',
    # 'Tradeoff',
]

COLS_METRICS_ACC = ["Accuracy", "Recall", "Precision", "F1", "Auc", "Rmse"]
COLS_METRICS_CTR = ["Click-Through-Rate", "Ndcg"]
COLS_METRICS_OTHER = ["Wilcoxon Signed Ranks Test", "Mrr", "Statistical-Test", "Cosine Similarity"]
COLS_METRICS_DIVERSITY = ["Intra-List-Diversity"]

COLS_VALUES = list(it.chain(
    COLS_USER_EXPERIENCE,
    COLS_NEWS_VALUES,
    COLS_NEWS_RECOMMENDER_SPECIFIC,
    COLS_RESPONSIBILITY,
    COLS_RESPONSIBILITY_AGENCY,
    COLS_GOTO,
    COLS_TECHNICAL,
))

COLS_VALUES_GRP = [
    C_GRP_GOTO,
    C_GRP_NEWS,
    C_GRP_RESPONSIBILITY,
    C_GRP_TECHNICAL,
    C_GRP_UX,
]
COLS_METRICS_GRP = [
    C_METRICS_ACC,
    C_METRICS_CTR,
    C_METRICS_OTHER,
    C_METRICS_DIVERSITY,
]
COLS_METRICS = [c.title() for c in list(it.chain(
    COLS_METRICS_ACC,
    COLS_METRICS_CTR,
    COLS_METRICS_OTHER,
    COLS_METRICS_DIVERSITY,
))]

COLS_ACCURACY = [c.title() for c in ['Click-Through-Rate', 'Accuracy', 'Precision', 'Recall', 'F1']]
molten_data = data.melt(id_vars=[C_TITLE, C_TYPE_PAPER, C_PLATFORM, C_YEAR] + COLS_METRICS, value_vars=COLS_VALUES, var_name=C_VALUE, value_name=C_MENTIONS)
molten_data[C_VALUE_GROUP] = ""
molten_data.loc[molten_data[C_VALUE].isin(COLS_USER_EXPERIENCE), C_VALUE_GROUP] = C_GRP_UX
molten_data.loc[molten_data[C_VALUE].isin(COLS_NEWS_VALUES + COLS_NEWS_RECOMMENDER_SPECIFIC), C_VALUE_GROUP] = C_GRP_NEWS
molten_data.loc[molten_data[C_VALUE].isin(COLS_RESPONSIBILITY + COLS_RESPONSIBILITY_AGENCY), C_VALUE_GROUP] = C_GRP_RESPONSIBILITY
molten_data.loc[molten_data[C_VALUE].isin(COLS_GOTO), C_VALUE_GROUP] = C_GRP_GOTO
molten_data.loc[molten_data[C_VALUE].isin(COLS_TECHNICAL), C_VALUE_GROUP] = C_GRP_TECHNICAL
molten_data = molten_data.sort_values([C_VALUE_GROUP, C_YEAR])
molten_data[C_METRICS_ACC] = molten_data[COLS_METRICS_ACC].sum(axis=1)
molten_data[C_METRICS_CTR] = molten_data[COLS_METRICS_CTR].sum(axis=1)
molten_data[C_METRICS_OTHER] = molten_data[COLS_METRICS_OTHER].sum(axis=1)
molten_data[C_METRICS_DIVERSITY] = molten_data[COLS_METRICS_DIVERSITY].sum(axis=1)
molten_data = molten_data.drop(COLS_METRICS, axis=1)
molten_data
# %%
corr_matrix = data[COLS_VALUES].corr()
corr_matrix
# %%
thresh = [0, 1, 3, 5, 10]
for i in thresh:
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_data = data.copy()
    non_unique_values = [col for col, decision in (corr_data[COLS_VALUES].sum(axis=0) > i).items() if decision]
    sns.heatmap(corr_data[non_unique_values].corr(), vmin=-1, vmax=1, cmap="bwr")
    fig.suptitle(f'Correlation of Values with at least {i} appearances.')
    fig.tight_layout()
    plt.savefig(f'figs/lit_rev_value_correlations_with_threshold_{i}.png')
    plt.show()
# %%
fig, ax = plt.subplots(figsize=(15, 8))
sns.barplot(
    data=molten_data.groupby([C_VALUE, C_VALUE_GROUP]).sum().reset_index().sort_values([C_VALUE_GROUP, C_MENTIONS]),
    y=C_VALUE,
    x=C_MENTIONS,
    hue=C_VALUE_GROUP,
    # estimator=sum,
    orient="h",
    ax=ax,
    ci=None,
    dodge=False,
    # linewidth=2.5,
    # facecolor=(1, 1, 1, 0),
    # errcolor=".2",
    # edgecolor=".2",
)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', padding=5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')
# ax.set_yticklabels(ax.get_yticklabels(), rotation=45, va='center')
fig.suptitle('Count of Value Contributions across the Literature')
fig.tight_layout()
plt.savefig('figs/lit_rev_count_value.png')
plt.show()

# %%
shortened_data = molten_data[molten_data[C_MENTIONS] > 0]
# pivoted_data = shortened_data.groupby([C_TITLE, C_TYPE_PAPER, C_VALUE_GROUP]).max().reset_index().groupby([C_TYPE_PAPER, C_VALUE_GROUP]).sum().reset_index().pivot(index=C_TYPE_PAPER, columns=C_VALUE_GROUP, values=C_MENTIONS)
pivoted_data = shortened_data.groupby([C_TYPE_PAPER, C_VALUE_GROUP]).sum().reset_index().pivot(index=C_TYPE_PAPER, columns=C_VALUE_GROUP, values=C_MENTIONS)
fig, ax = plt.subplots(figsize=(12, 6))
pivoted_data.plot(kind='bar', stacked=False, ax=ax)  #.set_index(C_TYPE_PAPER)[C_VALUE_GROUP].plot(kind='hist', stacked=True)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', padding=5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
fig.suptitle('Count of Value Contributions per Paper Type')
fig.tight_layout()
plt.savefig('figs/lit_rev_count_value_group_by_paper_type.png')
plt.show()

# %%
unique_values = pivoted_data.columns
unique_types = pivoted_data.index
num_rows = len(unique_types) // 2
num_cols = len(unique_types) // num_rows
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20), squeeze=False, sharex=True)

# for i, v in enumerate(unique_values):
for ax, t in zip(axes.flatten(), unique_types):
    tmp = pivoted_data.fillna(0).loc[t].to_frame()
    ax.pie(tmp[t], labels=tmp.index, autopct='%.0f%%')
    ax.set_title(f"Paper Type: {t}")
fig.suptitle('Share of Value Contributions per Paper Type')
fig.tight_layout()
plt.savefig('figs/lit_rev_count_paper_types_by_value_group_share.png')
plt.show()

# %%
pivoted_data_T = pivoted_data.T
fig, ax = plt.subplots(figsize=(12, 6))
pivoted_data_T.plot(kind='bar', stacked=False, ax=ax)  #.set_index(C_TYPE_PAPER)[C_VALUE_GROUP].plot(kind='hist', stacked=True)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', padding=5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(loc='upper right')
fig.suptitle('Count of Paper Types per Value Contribution')
fig.tight_layout()
plt.savefig('figs/lit_rev_count_paper_types_by_value_group.png')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(15, 10))
sns.countplot(
    data=data,
    x=C_TYPE_PAPER,
    # estimator=sum,
    # orient="h",
    ax=ax,
    # ci=None,
    # linewidth=2.5,
    facecolor=(0.5, 0.5, 0.5, 1),
    # edgecolor=".2",
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='center')
ax.bar_label(ax.containers[0], label_type='edge', padding=5)

fig.suptitle('Count of Paper Types')
fig.tight_layout()
plt.savefig('figs/lit_rev_count_paper_types.png')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(12, 10))
sns.countplot(
    data=data,
    y=C_PLATFORM,
    # estimator=sum,
    # orient="h",
    ax=ax,
    # ci=None,
    linewidth=2.5,
    facecolor=(1, 1, 1, 0),
    edgecolor=".2",
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='center')
ax.bar_label(ax.containers[0], label_type='edge', padding=5)
fig.suptitle('Count of Platforms')
plt.savefig('figs/lit_rev_count_platforms.png')
plt.show()

# %%
grouped_time_data = molten_data.groupby([C_YEAR, C_VALUE_GROUP]).sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.lineplot(
    data=grouped_time_data[grouped_time_data[C_YEAR] <= data[C_YEAR].max() - pd.DateOffset(years=1)],
    x=C_YEAR,
    y=C_MENTIONS,
    hue=C_VALUE_GROUP,
    ax=ax,
)
ax = sns.lineplot(
    data=grouped_time_data[grouped_time_data[C_YEAR] >= data[C_YEAR].max() - pd.DateOffset(years=1)],
    x=C_YEAR,
    y=C_MENTIONS,
    hue=C_VALUE_GROUP,
    ax=ax,
    linestyle="-.",
    legend=None,
)

sns.lineplot(data=data[[C_YEAR, C_TITLE]].groupby(C_YEAR).count(), x=C_YEAR, y=C_TITLE, label="No. of relevant Papers published", color='Grey', linestyle=":")
# sns.lineplot(data=df_all_pre_screening[[C_YEAR, C_TITLE]].groupby(C_YEAR).count(), x=C_YEAR, y=C_TITLE, label="No. of relevant Papers published", color='Grey', linestyle="-.")
# sns.lineplot(data=grouped_time_data.groupby(C_YEAR).mean(), x=C_YEAR, y=C_MENTIONS, label="Mean Value Contributions", color='Black', linestyle="-.")
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=2))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
# ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
# ax.tick_params(axis='y', which='minor', bottom=False)
# ax.set_xlim(data[C_YEAR].min(), data[C_YEAR].max() - pd.DateOffset(years=1))
ax.grid(axis='y', )
fig.suptitle("Sum of Value Contributions over Time")
fig.tight_layout()
plt.savefig('figs/lit_rev_count_over_time.png')
plt.show()
# %%
grouped_time_data = molten_data.groupby([C_YEAR, C_VALUE_GROUP]).sum().reset_index().merge(data[[C_YEAR, C_TITLE]].groupby(C_YEAR).count().reset_index(),
                                                                                           left_on=C_YEAR,
                                                                                           right_on=C_YEAR)
grouped_time_data[C_MENTIONS] /= grouped_time_data[C_TITLE]
fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.lineplot(
    data=grouped_time_data[grouped_time_data[C_YEAR] <= data[C_YEAR].max() - pd.DateOffset(years=1)],
    x=C_YEAR,
    y=C_MENTIONS,
    hue=C_VALUE_GROUP,
    ax=ax,
)
ax = sns.lineplot(
    data=grouped_time_data[grouped_time_data[C_YEAR] >= data[C_YEAR].max() - pd.DateOffset(years=1)],
    x=C_YEAR,
    y=C_MENTIONS,
    hue=C_VALUE_GROUP,
    ax=ax,
    linestyle="-.",
    legend=None,
)

# sns.lineplot(data=, x=C_YEAR, y=C_TITLE, label="No. of relevant Papers published", color='Grey', linestyle="-.")
# sns.lineplot(data=grouped_time_data.groupby(C_YEAR).mean(), x=C_YEAR, y=C_MENTIONS, label="Mean Value Contributions", color='Black', linestyle="-.")
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=2))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
ax.grid(axis='y', )
fig.suptitle("Sum of Value Contribution Proportions over Time")
fig.tight_layout()
plt.savefig('figs/lit_rev_count_proportions_over_time.png')
plt.show()
# %%
# grouped_time_data = molten_data.groupby([C_YEAR, C_VALUE_GROUP]).sum().reset_index()
# fig, ax = plt.subplots(figsize=(12, 6))
# ax = sns.lineplot(
#     data=grouped_time_data,
#     x=C_YEAR,
#     y=C_MENTIONS,
#     hue=C_VALUE_GROUP,
#     ax=ax,
# )

# sns.lineplot(data=data[[C_YEAR, C_TITLE]].groupby(C_YEAR).count(), x=C_YEAR, y=C_TITLE, label="No. of relevant Papers published", color='Grey', linestyle="-.")
# # sns.lineplot(data=df_all_pre_screening[[C_YEAR, C_TITLE]].groupby(C_YEAR).count(), x=C_YEAR, y=C_TITLE, label="No. of relevant Papers published", color='Grey', linestyle="-.")
# # sns.lineplot(data=grouped_time_data.groupby(C_YEAR).mean(), x=C_YEAR, y=C_MENTIONS, label="Mean Value Contributions", color='Black', linestyle="-.")
# ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=2))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
# # ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
# # ax.tick_params(axis='y', which='minor', bottom=False)
# ax.set_xlim(data[C_YEAR].min(), data[C_YEAR].max() - pd.DateOffset(years=1))
# ax.grid(axis='y', )
# fig.suptitle("Sum of Value Contributions over Time (until 2021)")
# fig.tight_layout()
# plt.savefig('figs/lit_rev_count_over_time_only_completed_years.png')
# plt.show()
# %%
grouped_time_data = molten_data.groupby([C_YEAR]).sum().div(36).reset_index().melt([C_YEAR], COLS_METRICS_GRP, value_name=C_MENTIONS, var_name=C_METRICS_GROUP)
fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.lineplot(
    data=grouped_time_data[grouped_time_data[C_YEAR] <= data[C_YEAR].max() - pd.DateOffset(years=1)],
    x=C_YEAR,
    y=C_MENTIONS,
    hue=C_METRICS_GROUP,
    ax=ax,
)
ax = sns.lineplot(
    data=grouped_time_data[grouped_time_data[C_YEAR] >= data[C_YEAR].max() - pd.DateOffset(years=1)],
    x=C_YEAR,
    y=C_MENTIONS,
    hue=C_METRICS_GROUP,
    ax=ax,
    linestyle="-.",
    legend=None,
)

ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=2))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
# ax.set_xlim(data[C_YEAR].min(), data[C_YEAR].max() - pd.DateOffset(years=1))
ax.grid(axis='y', )
fig.suptitle("Sum of Metric Uses over Time (until 2021)")
fig.tight_layout()
plt.savefig('figs/lit_rev_metrics_count_over_time_only_completed_years.png')
plt.show()

# %%
df_all_pre_screening = pd.read_csv('data_literature_review/data_pre_screening.csv')
df_all_pre_screening[C_YEAR] = pd.to_datetime(df_all_pre_screening["year"].astype(str), format="%Y")
df_all_pre_screening[C_TITLE] = df_all_pre_screening["title"]
df_all_pre_screening = df_all_pre_screening[df_all_pre_screening[C_YEAR] >= data[C_YEAR].min()]
df_all_pre_screening
# grouped_time_data = molten_data.groupby([C_YEAR, C_VALUE_GROUP]).sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
# ax = sns.lineplot(
#     data=grouped_time_data,
#     x=C_YEAR,
#     y=C_MENTIONS,
#     hue=C_VALUE_GROUP,
#     ax=ax,
# )

df_pre_screening = df_all_pre_screening[[C_YEAR, C_TITLE]].groupby(C_YEAR).count()
df_post_screening = data[[C_YEAR, C_TITLE]].groupby(C_YEAR).count()
ax = sns.lineplot(
    data=df_pre_screening[df_pre_screening.index <= data[C_YEAR].max() - pd.DateOffset(years=1)],
    x=C_YEAR,
    y=C_TITLE,
    color="blue",
    label="No. of News Recommeder Papers published",
    ax=ax,
)
ax = sns.lineplot(
    data=df_pre_screening[df_pre_screening.index >= data[C_YEAR].max() - pd.DateOffset(years=1)],
    x=C_YEAR,
    y=C_TITLE,
    color="blue",
    # label="No. of News Recommeder Papers published",
    linestyle="-.",
    legend=None,
    ax=ax,
)
ax = sns.lineplot(
    data=df_post_screening[df_post_screening.index <= data[C_YEAR].max() - pd.DateOffset(years=1)],
    x=C_YEAR,
    y=C_TITLE,
    color="red",
    label="No. of Value-Driven News Recommender Papers published",
    ax=ax,
)
ax = sns.lineplot(
    data=df_post_screening[df_post_screening.index >= data[C_YEAR].max() - pd.DateOffset(years=1)],
    x=C_YEAR,
    y=C_TITLE,
    color="red",
    # label="No. of Value-Driven News Recommender Papers published",
    linestyle="-.",
    legend=None,
    ax=ax,
)

# sns.lineplot(data=grouped_time_data.groupby(C_YEAR).mean(), x=C_YEAR, y=C_MENTIONS, label="Mean Value Contributions", color='Black', linestyle="-.")
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=2))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
ax.grid(axis='y')
fig.suptitle("Count of Value Contributing Papers compared to News RS literature over Time")
fig.tight_layout()
plt.savefig('figs/lit_rev_count_nrs_over_time.png')
plt.show()
# %%
df_all_pre_screening = pd.read_csv('data_literature_review/data_pre_screening.csv')
df_all_pre_screening[C_YEAR] = pd.to_datetime(df_all_pre_screening["year"].astype(str), format="%Y")
df_all_pre_screening[C_TITLE] = df_all_pre_screening["title"]
df_all_pre_screening = df_all_pre_screening[df_all_pre_screening[C_YEAR] >= data[C_YEAR].min()]
df_all_pre_screening

grouped_time_data = molten_data.groupby([C_YEAR, C_VALUE_GROUP]).sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 6))

proportion_data = ((data[[C_YEAR, C_TITLE]].groupby(C_YEAR).count() / df_all_pre_screening[[C_YEAR, C_TITLE]].groupby(C_YEAR).count())) * 100
ax = sns.lineplot(
    data=proportion_data[proportion_data.index <= data[C_YEAR].max() - pd.DateOffset(years=1)],
    x=C_YEAR,
    y=C_TITLE,
    color="blue",
    label="Proportion of Value-Driven News Recommender Papers published",
    ax=ax,
)
ax = sns.lineplot(
    data=proportion_data[proportion_data.index >= data[C_YEAR].max() - pd.DateOffset(years=1)],
    x=C_YEAR,
    y=C_TITLE,
    color="blue",
    # label="Proportion of Value-Driven News Recommender Papers published",
    linestyle="-.",
    legend=None,
    ax=ax,
)
# sns.lineplot(data=grouped_time_data.groupby(C_YEAR).mean(), x=C_YEAR, y=C_MENTIONS, label="Mean Value Contributions", color='Black', linestyle="-.")
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=2))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
ax.grid(axis='y')
ax.set_ylabel('Percentage of value contributing papers against all news recommender paper')
fig.suptitle("Proportion of Value Contributing Papers compared to News RS literature over Time")
fig.tight_layout()
plt.savefig('figs/lit_rev_count_proportions_over_time.png')
plt.show()

# %%
# fig, ax = plt.subplots(figsize=(12, 6))

# %%


# https://deffro.github.io/time%20series/exploratory%20data%20analysis/data%20visualization/time-series-analysis/
def helper_fn(df):
    # print(df)
    df[C_ROLLING] = df[C_MENTIONS].ewm(halflife="3 Y", times=pd.DatetimeIndex(df[C_YEAR])).mean()
    return df


rolling_time_data = grouped_time_data.groupby([C_VALUE_GROUP]).apply(helper_fn)
fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.lineplot(
    data=rolling_time_data,
    x=C_YEAR,
    y=C_ROLLING,
    hue=C_VALUE_GROUP,
    # estimator=sum,
    ax=ax,
)
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=1))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
ax.set_xlim(data[C_YEAR].min(), data[C_YEAR].max() - pd.DateOffset(years=1))
ax.grid(axis='y', )
sns.lineplot(data=rolling_time_data.groupby(C_YEAR).mean(), x=C_YEAR, y=C_ROLLING, label="Mean", color='Black', linestyle="-.")
fig.suptitle("Trend of Value Contributions over Time (using EWMA)")
fig.tight_layout()
plt.savefig('figs/lit_rev_count_over_time_rolling.png')
plt.show()


# %%
# pivoted_data_grp = molten_data[[C_TITLE, C_YEAR, C_VALUE,C_VALUE_GROUP, C_MENTIONS]].pivot(index=[C_TITLE, C_YEAR], columns=[C_VALUE_GROUP], values=[C_MENTIONS])
# https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
# https://stackoverflow.com/a/32106967
def helper_timewise_correlation(df):
    return df.corr()


# data[data[C_TITLE] == 'Customized Internet News Services Based on Customer Profiles']
def helper_timewise_cooccurrence(df):
    tmp = df.drop([C_TITLE, C_YEAR], axis=1)
    X = tmp.values
    new_df = pd.DataFrame(X.T @ X, columns=tmp.columns, index=tmp.columns)
    return new_df


pivoted_data_grp = molten_data.groupby([C_YEAR, C_TITLE, C_VALUE_GROUP]).sum().reset_index().pivot(index=[C_TITLE, C_YEAR], columns=[C_VALUE_GROUP],
                                                                                                   values=[C_MENTIONS]).reset_index().sort_values(C_YEAR, ascending=False)
pivoted_data_grp_cooc = pivoted_data_grp.groupby(C_YEAR).apply(helper_timewise_cooccurrence).droplevel(0, axis=1).droplevel(1, axis=0).reset_index()
pivoted_data_grp_cooc.columns.name = ""
pivoted_data_grp_cooc

# %%
pairwise_data_grp_cooc = pivoted_data_grp_cooc.melt(id_vars=[C_YEAR, C_VALUE_GROUP], value_vars=COLS_VALUES_GRP, var_name=C_COMPLEMENT, value_name=C_COOC)
# pairwise_data_grp_cooc = pd.melt(pivoted_data_grp_cooc, id_vars=[C_YEAR, C_VALUE_GROUP], value_vars=COLS_VALUES_GRP, ignore_index=True)
pairwise_data_grp_cooc[C_COMBINATION] = pairwise_data_grp_cooc[[C_VALUE_GROUP, C_COMPLEMENT]].agg(' - '.join, axis=1)
pairwise_data_grp_cooc
# %%
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(
    data=pairwise_data_grp_cooc[pairwise_data_grp_cooc[C_VALUE_GROUP] != pairwise_data_grp_cooc[C_COMPLEMENT]],
    x=C_YEAR,
    y=C_COOC,
    hue=C_COMBINATION,
)
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=2))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
ax.grid(axis='y', )
fig.suptitle("Co-Occurences of Value Contributions over Time")
fig.tight_layout()
plt.savefig('figs/lit_rev_cooc_over_time.png')
plt.show()
# %%
# fig, ax = plt.subplots(figsize=(12, 6))
fig = sns.relplot(
    data=pairwise_data_grp_cooc[pairwise_data_grp_cooc[C_VALUE_GROUP] != pairwise_data_grp_cooc[C_COMPLEMENT]],
    x=C_YEAR,
    y=C_COOC,
    col=C_VALUE_GROUP,
    hue=C_COMPLEMENT,
    kind="line",
    col_wrap=2,
)
# ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=2))
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
fig.figure.suptitle("Co-Occurences of Value Contributions over Time for each Value Group")
fig.tight_layout()
plt.savefig('figs/lit_rev_cooc_over_time_subplots.png')
plt.show()
# %%
data[COLS_METRICS].values.shape
# %%
coocc = data[COLS_VALUES].values.T @ data[COLS_METRICS].values
coocc
# %%
# TODO: Maybe think of a weighting approach like tf-idf to also incoporate overall occurence
# TODO: Maybe use a better color palette.
plt.figure(figsize=(15, 10))
tmp = pd.DataFrame(coocc, columns=COLS_METRICS, index=COLS_VALUES).T
sns.heatmap(tmp)
plt.savefig('figs/lit_rev_metric_v_values_cooc.png')
plt.show()
# %%
plt.figure(figsize=(15, 10))
normed_coocc = coocc / data[COLS_VALUES].sum().values[:, None]
tmp = pd.DataFrame(normed_coocc, columns=COLS_METRICS, index=COLS_VALUES).T
sns.heatmap(tmp)
plt.savefig('figs/lit_rev_metric_v_values_cooc_normed_by_values.png')
plt.show()
# %%
plt.figure(figsize=(15, 10))
normed_coocc_T = (coocc.T / data[COLS_METRICS].sum().values[:, None]).T
tmp = pd.DataFrame(normed_coocc_T, columns=COLS_METRICS, index=COLS_VALUES).T
sns.heatmap(tmp)
plt.savefig('figs/lit_rev_metric_v_values_cooc_normed_by_metric.png')
plt.show()
# %%
plt.figure(figsize=(12, 8))
all_grouped_data = molten_data.pivot(
    index=[
        C_TITLE,
        # C_YEAR,
        C_VALUE,
        C_METRICS_ACC,
        C_METRICS_CTR,
        C_METRICS_DIVERSITY,
        C_METRICS_OTHER,
    ],
    columns=C_VALUE_GROUP,
    values=C_MENTIONS).reset_index().fillna(0).groupby(C_TITLE).max().corr()
all_grouped_data
sns.heatmap(all_grouped_data, vmin=-1, vmax=1, cmap="bwr")
plt.savefig('figs/lit_rev_metric_v_values_group_corr.png')
plt.show()
# %%

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), squeeze=True)
exploded_data = data[[C_YEAR, C_TITLE, C_DIVERSITY_TYPE]]
exploded_data = exploded_data.explode(C_DIVERSITY_TYPE)
tmp_counts = pd.value_counts(exploded_data[C_DIVERSITY_TYPE])
sns.lineplot(data=exploded_data.groupby([C_YEAR, C_DIVERSITY_TYPE]).count(), x=C_YEAR, y=C_TITLE, hue=C_DIVERSITY_TYPE, ax=ax1)
sns.countplot(exploded_data[C_DIVERSITY_TYPE], order=tmp_counts[tmp_counts > 1].index, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='center')
fig.suptitle("Diversity-Type over Time and in absolute Counts")
fig.tight_layout()
plt.savefig('figs/lit_rev_diversity_type_counts_double_plot.png')
# %%
fig, ax = plt.subplots(1, 1, figsize=(12, 6), squeeze=True)
sns.countplot(x=exploded_data[C_DIVERSITY_TYPE], order=tmp_counts[tmp_counts >= 1].index, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
ax.bar_label(ax.containers[0], label_type='edge', padding=5)
fig.suptitle("Diversity-Type in absolute Counts")
fig.tight_layout()
plt.savefig('figs/lit_rev_diversity_type_counts_all.png')
# %%
mini_counts = (exploded_data[C_DIVERSITY_TYPE].value_counts() < 2)
summarised_exploded_data = exploded_data.copy()
cols_to_group = mini_counts[mini_counts == True].index
summarised_exploded_data[summarised_exploded_data[C_DIVERSITY_TYPE].isin(cols_to_group)] = "Other"
fig, ax = plt.subplots(1, 1, figsize=(12, 6), squeeze=True)
sns.countplot(x=summarised_exploded_data[C_DIVERSITY_TYPE], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
ax.bar_label(ax.containers[0], label_type='edge', padding=5)
fig.suptitle("Diversity-Type in absolute Counts")
fig.tight_layout()
plt.savefig('figs/lit_rev_diversity_type_counts_more_than_1.png')
plt.show()
# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), squeeze=True)
exploded_data = data[[C_YEAR, C_TITLE, C_EVALUATION_TYPE]]
exploded_data = exploded_data.explode(C_EVALUATION_TYPE)
tmp_counts = pd.value_counts(exploded_data[C_EVALUATION_TYPE])
sns.lineplot(data=exploded_data.groupby([C_YEAR, C_EVALUATION_TYPE]).count(), x=C_YEAR, y=C_TITLE, hue=C_EVALUATION_TYPE, ax=ax1)
sns.countplot(exploded_data[C_EVALUATION_TYPE], ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='center')
fig.suptitle("Evaluation-Type over Time and in absolute Counts")
fig.tight_layout()
plt.savefig('figs/lit_rev_evaluation_type_counts_double_plot.png')
# %%
