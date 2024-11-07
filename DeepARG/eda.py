#%%
import pandas as pd
from Bio import SeqIO
# %%
def fasta_to_dataframe(fasta_file):
    records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        records.append({"id": record.id.split('|')[0]
                        , "db": record.id.split('|')[2]
                        , "type": record.id.split('|')[3]
                        , "sequence": str(record.seq)})
    return pd.DataFrame(records)

df = fasta_to_dataframe("../data/database/v1/features.fasta")
#%%
df.db.value_counts()

# %%
df['type'].value_counts()
# %%
df['type'].nunique()

# %%
df.sequence.apply(lambda x:len(x)).describe()
# %%
import json
from pprint import pprint
# %%
with open("../data/scripts/db/FilteredDatabase.json") as data:
    d = json.load(data)
    data.close()
    pprint(d)
# %%
len(d)
# %%
import pickle
# %%
meta_data = pickle.load("../data/model/v1/metadata_SS.pkl")

# %%
