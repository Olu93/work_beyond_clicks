# %%
from pybtex.database import parse_file, Entry, Person, BibliographyData
from pybtex import errors
import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
errors.set_strict_mode(False)

# %%
def extract_relevant_features(entry: Entry):
    result = {}
    result['title'] = entry.fields.get('title', "")
    result['journal'] = entry.fields.get('journal', "")
    result['doi'] = entry.fields.get('doi', "").replace("{", "").replace("}", "")
    
    result['authors'] = "; ".join([str(person) for person in entry.persons['author']]) if "author" in entry.persons else ""
    return result


def extract_bibtex_info(name: str):
    db: BibliographyData = parse_file(f"bibtex/{name}.bib", strict=False)
    return [extract_relevant_features(db.entries[entry_name]) for entry_name in db.entries]

def extract_springer_info(name: str):
    db = pd.read_csv(f'csv/{name}.csv')[["Item Title", "Publication Title", "Item DOI", "Authors"]]
    db.columns = ["title", "journal", "doi", "authors"]
    return db

db_scopus = pd.DataFrame(extract_bibtex_info("scopus"))
db_wos = pd.DataFrame(extract_bibtex_info("wos"))
db_acm = pd.DataFrame(extract_bibtex_info("acm"))
db_ieee = pd.DataFrame(extract_bibtex_info("ieee"))
db_springer = extract_springer_info("springer")


all_dbs = {"scopus": db_scopus, "Web of Science": db_wos, "ACM": db_acm, "IEEE": db_ieee, "Springer": db_springer}
for name, db in all_dbs.items():
    print(name)
    display(db)
# %%
all_info = []
for k1, v1 in all_dbs.items():
    for k2, v2 in all_dbs.items():
        intersections = len(set(v1.doi).intersection(set(v2.doi)))
        all_info.append({"db1":k1, "db2":k2, "intersections":intersections/len(set(v1.doi)), "duplicates": intersections})

df_all_info = pd.DataFrame(all_info)
df_all_info
# %%
plt.figure(figsize=(10,7))
intersections_heatmap = df_all_info.pivot("db1", "db2", "intersections")
ax = sns.heatmap(intersections_heatmap, annot=True)
ax.set_title("Amount of intersections normalized by size of DB1.\n'news recommender' as search query in Title, Abstract and Keywords")
# %%
plt.figure(figsize=(10,7))
intersections_heatmap = df_all_info.pivot("db1", "db2", "duplicates")
ax = sns.heatmap(intersections_heatmap, annot=True, fmt="d")
ax.set_title("Duplicates count.\n'news recommender' as search query in Title, Abstract and Keywords")

# %%
