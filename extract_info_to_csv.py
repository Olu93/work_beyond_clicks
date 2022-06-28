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


db_selected_papers = pd.DataFrame(extract_bibtex_info("selected_papers"))

db_selected_papers
