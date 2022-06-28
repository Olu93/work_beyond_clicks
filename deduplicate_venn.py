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
    db: BibliographyData = parse_file(f"relevant/{name}.bib", strict=False)
    return [extract_relevant_features(db.entries[entry_name]) for entry_name in db.entries]

# %%
db_scopus = pd.DataFrame(extract_bibtex_info("scopus"))
db_springer = pd.DataFrame(extract_bibtex_info("springer"))
# %%
num_duplicates_scopus = db_scopus[db_scopus.duplicated(subset='doi')]
display(num_duplicates_scopus)
set_duplicates_scopus=set(num_duplicates_scopus.doi)
len(set_duplicates_scopus)
# %%
num_duplicates_springer=db_springer[db_springer.duplicated(subset='doi')]
display(num_duplicates_springer)
set_duplicates_springer=set(num_duplicates_springer.doi)
len(set_duplicates_springer)
# %%
combined = pd.concat([db_scopus.drop_duplicates(subset='doi'), db_springer.drop_duplicates(subset='doi')])
num_duplicates_combined = combined[combined.duplicated(subset='doi')]
display(num_duplicates_combined)
set_duplicates_combined=set(num_duplicates_combined.doi)
len(set_duplicates_combined)
# %%
raw = pd.concat([db_scopus, db_springer])
num_duplicates_raw = raw[raw.duplicated(subset='doi')]
display(num_duplicates_raw)
set_duplicates_raw=set(num_duplicates_raw.doi)
len(set_duplicates_raw)
# # %%
# from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
# venn3(subsets=(len(set_duplicates_raw), len(set_duplicates_scopus), len(set_duplicates_springer)))
# %%
combined.to_csv('csv/combined_database.csv')