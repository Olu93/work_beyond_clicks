# %%
from pybtex.database import parse_file, Entry, Person, BibliographyData
from pybtex import errors
import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
errors.set_strict_mode(False)

# %%
# pd.read_csv(f'csv/springer.csv')


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
    tmp = pd.read_csv(f'csv/{name}.csv')
    db = tmp[[
        "Item Title",
        "Publication Title",
        "Item DOI",
        "Authors",
        "Publication Year",
        "URL",
        "Content Type",
        "Journal Volume",
        "Journal Issue"
    ]]
    db.columns = ["title", "journal", "doi", "authors", "year", "url", "document_type", "volume", "number"]
    db.document_type[db.document_type=="Chapter"] = "incollection"
    db.document_type[db.document_type=="Book"] = "book"
    db.document_type[db.document_type=="Article"] = "article"
    db.volume = db.volume.astype(str).str.replace(".0", "")
    return db


db_springer = pd.concat([
    extract_springer_info("springer_news_personali"),
    extract_springer_info("springer_news_recommend"),
    extract_springer_info("springer_personali_news"),
])
db_springer["id"] = ""
db_springer["id"] = db_springer["id"] + db_springer.year.astype(str)  + "_" + db_springer.document_type + "_" + [f"{val:04d}" for val in db_springer.index]

db_springer
# %% 

prep_2 = {}
for obj in db_springer.to_dict('records'):
    idx = obj["id"]
    authors_field = obj["authors"]
    # print(authors_field)
    persons = []
    if authors_field is not np.nan:
        authors = re.split("(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", authors_field)
        persons = [Person(name) for name in authors]
    document_type = obj["document_type"]
    del obj["id"]
    del obj["document_type"]
    del obj["authors"]
    prep_2[idx] = Entry(document_type, [(k, str(v)) for k, v in obj.items() if str(v) != "nan"])
    if len(persons) > 0: 
        prep_2[idx].persons['author'] = persons 
        
list(prep_2.items())[:5]
# %%
bibtex_data = BibliographyData(prep_2)
# %%
with open('springer.bib', 'w', encoding='utf-8') as f:
    f.write(bibtex_data.to_string('bibtex'))
# %%
