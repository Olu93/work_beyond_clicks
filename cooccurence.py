# %%
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

#%%

data = pd.read_csv(io.open('literature_review_merged_karin_chandni.csv', 'r', encoding='iso-8859-1'))
data = data.set_index("ID")
data = data[data['Not Relevant'] != "X"]
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# %%
all_relevant_cols = [c for c in data.columns if "X" in data[c].unique()]
all_relevant_cols
# %%
# %%
data[all_relevant_cols] = data[all_relevant_cols]
data[all_relevant_cols] = data[all_relevant_cols].replace(["X", "x"], 1)
data[all_relevant_cols] = data[all_relevant_cols].replace([""], 0)
data[all_relevant_cols] = data[all_relevant_cols].fillna(0)
data[all_relevant_cols] = data[all_relevant_cols].infer_objects()
data[all_relevant_cols] = data[all_relevant_cols].astype(float)
data[all_relevant_cols].dtypes
# %%
# %%
value_cols = [
    'Diversity3', 'Agency3', 'Novelty3', 'Context', 'Explainability3', 'Democracy3', 'Privacy3', 'Accountability3', 'Trust3', 'Transparency3', 'Objectivity3', 'Serendipity3',
    'Location3', 'Recency3', 'Coverage3', 'Newsworthiness/Relevance', 'Emotion3', 'Interests over time (short medium long)', 'Decentralization', 'Engagement',
    'Temporality of interests', 'Fairness'
] 

metric_cols = ['Click-Through-Rate3', 'Accuracy3', 'Precision3', 'Recall3', 'F1_3', 'Intra-list-diversity3', 'Cosine similarity3', 'Wilcoxon Signed Ranks Test3', 'nDCG3'] 
accuracy_cols = ['Click-Through-Rate3', 'Accuracy3', 'Precision3', 'Recall3', 'F1_3']


# %%
data[value_cols].corr()
# %%
sns.heatmap(data[value_cols].corr())
plt.show()
# %%
data[value_cols].values.shape
# %%
data[metric_cols].values.shape
# %%
coocc = data[value_cols].values.T @ data[metric_cols].values
coocc
# %%
tmp = pd.DataFrame(coocc, columns=metric_cols, index=value_cols).T
sns.heatmap(tmp)
plt.show()
# %%
normed_coocc = coocc/data[value_cols].sum().values[:, None]
tmp = pd.DataFrame(normed_coocc, columns=metric_cols, index=value_cols).T
sns.heatmap(tmp)
plt.show()
# %%
normed_coocc_T = (coocc.T/data[metric_cols].sum().values[:, None]).T
tmp = pd.DataFrame(normed_coocc_T, columns=metric_cols, index=value_cols).T
sns.heatmap(tmp)
plt.show()
# %%
