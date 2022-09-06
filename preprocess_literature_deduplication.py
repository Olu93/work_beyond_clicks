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
    db: BibliographyData = parse_file(f"data_literature_search/relevant/{name}.bib", strict=False)
    return [extract_relevant_features(db.entries[entry_name]) for entry_name in db.entries]

# %%
db_scopus = pd.DataFrame(extract_bibtex_info("scopus"))
db_springer = pd.DataFrame(extract_bibtex_info("springer"))
# %%
num_duplicates_scopus = db_scopus.duplicated(subset='doi').sum()
num_duplicates_scopus
# %%
num_duplicates_springer=db_springer.duplicated(subset='doi').sum()
num_duplicates_springer
# %%
combined = pd.concat([db_scopus.drop_duplicates(subset='doi'), db_springer.drop_duplicates(subset='doi')])
num_duplicates_combined = combined.duplicated(subset='doi').sum()
num_duplicates_combined
# %%
raw = pd.concat([db_scopus, db_springer])
num_duplicates_raw = raw.duplicated(subset='doi').sum()
num_duplicates_raw
# %%
combined.to_csv('data_literature_search/csv/combined_database.csv')
combined.to_csv('data_literature_review/data_pre_screening.csv')
# %%
db_scopus