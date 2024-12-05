#%%
import pandas as pd
from Bio import Entrez, SeqIO
from collections import defaultdict
import math
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import concurrent.futures
from tqdm import tqdm
from itertools import repeat
import logging

Entrez.email = "matthew.mollerus@gmail.com"
Entrez.api_key = "3826be0f01d8949a3a47931277f819a5c208"
# %%
def fasta_to_dataframe(fasta_file):
    records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        records.append({"id": record.id.split('|')[0]
                        , "db": record.id.split('|')[2]
                        , "type": record.id.split('|')[3]
                        , "sequence": str(record.seq)})
    return pd.DataFrame(records)

aa_df = fasta_to_dataframe("../data/database/v1/features.fasta")
# %%
acquired_uniprot_df = pd.read_csv('uniprot_dna_seq.csv')
acquired_non_uniprot_df = pd.read_csv('non_uniprot_dna_seq.csv')

# %%
missing_up = acquired_uniprot_df[acquired_uniprot_df.dna_seq == '[Error] Fetching GenBank record []: list index out of range']
# %%
def fetch_uniprot_entry(uniprot_acc):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_acc}.json"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        # print(f"Failed to fetch data for {uniprot_acc}: Status Code {response.status_code}")
        return f"Failed to fetch data for {uniprot_acc}: Status Code {response.status_code}"

def extract_uniparc_id(uniprot_entry):
    try:
        return uniprot_entry['extraAttributes']['uniParcId']
    except KeyError:
        print("uniParcId not found in UniProt entry.")
        return None
# %%
def fetch_uniparc_entry(uniparc_id):
    url = f"https://rest.uniprot.org/uniparc/{uniparc_id}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching UniParc entry for {uniparc_id}: {e}")
        return None

def extract_emblwgs_ids(uniparc_entry):
    emblwgs_ids = []
    try:
        for db_ref in uniparc_entry['uniParcCrossReferences']:
            if (db_ref['database'] == 'EMBLWGS') | (db_ref['database'] == 'EMBL'):
                emblwgs_ids.append(db_ref.get('id'))
    except KeyError:
        print("Error extracting EMBL-WGS IDs from UniParc entry.")
    return emblwgs_ids

#%%
def fetch_ena_sequence(emblwgs_id):
    url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{emblwgs_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        fasta = response.text
        # Parse FASTA to extract the sequence
        lines = fasta.strip().split('\n')
        sequence = ''.join(lines[1:])  # Skip the header
        return sequence
    except requests.RequestException as e:
        print(f"Error fetching DNA sequence for {emblwgs_id}: {e}")
        return None
    
#%%
def process_uniprot_id(uniprot_acc):
    try:
        up_entry = fetch_uniprot_entry(uniprot_acc=uniprot_acc)
        if not up_entry:
            print(f"UniProt entry not found for ID {uniprot_acc}")
            return None

        uparc_id = extract_uniparc_id(up_entry)
        if not uparc_id:
            print(f"UniParc ID extraction failed for UniProt ID {uniprot_acc}")
            return None

        uparc_entry = fetch_uniparc_entry(uparc_id)
        if not uparc_entry:
            print(f"UniParc entry not found for UniParc ID {uparc_id}")
            return None

        ena_ids = extract_emblwgs_ids(uparc_entry)
        if not ena_ids:
            print(f"No EMBL-WGS IDs found for UniParc ID {uparc_id}")
            return None

        # Fetch DNA sequence using the first EMBL-WGS ID
        ena_seq = fetch_ena_sequence(ena_ids[0])
        if not ena_seq:
            print(f"DNA sequence not found for EMBL-WGS ID {ena_ids[0]}")
            return None

        return ena_seq

    except Exception as e:
        print(f'Failed to get sequence for UniProt ID {uniprot_acc}: {e}')
        return None

#%%
def parallel_fetch_dna_sequences(missing_up, max_workers=6):
    dna_seqs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use list to ensure the order is preserved
        futures = {executor.submit(process_uniprot_id, acc): acc for acc in missing_up.id.values}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing IDs"):
            acc = futures[future]
            try:
                seq = future.result()
                dna_seqs.append(seq)
            except Exception as e:
                print(f'Failed to get sequence for UniProt ID {acc}: {e}')
                dna_seqs.append(None)

    return dna_seqs

#%%
dna_seq = parallel_fetch_dna_sequences(missing_up, max_workers=6)

# %%
dna_seqs = []
for i,id in enumerate(missing_up.id.values):
    if i %100 == 0:
        print(i)
    try:
        up_entry = fetch_uniprot_entry(uniprot_acc = id)
        uparc_id = extract_uniparc_id(up_entry)
        uparc_entry = fetch_uniparc_entry(uparc_id)
        ena_ids = extract_emblwgs_ids(uparc_entry)
        ena_seq = fetch_ena_sequence(ena_ids[0])
        dna_seqs.append(ena_seq)
    except:
        print(f'failed to get seq for id {id}')
        dna_seqs.append(None)

# %%
len(dna_seq)
# %%
missing_up['dna_seq'] = dna_seq
# %%
# missing_up.to_csv('new_uniport_dna_seq.csv')
missing_up = pd.read_csv('new_uniport_dna_seq.csv')
# %%
missing_up[missing_up['dna_seq'].apply(lambda x:type(x)) == str]
# %%
len(missing_up)
# %%









#%%
missing_nup = acquired_non_uniprot_df[~(acquired_non_uniprot_df.dna_seq.apply(lambda x:type(x)) == str)]
# %%
missing_nup

#%%
def get_dna_sequence_from_protein(protein_id,aa_seq):
    try:
        # Search for the protein record and get the linked nucleotide ID
        handle = Entrez.elink(dbfrom="protein", id=protein_id, db="nucleotide")
        record = Entrez.read(handle)
        handle.close()
        
        # Extract the linked nucleotide ID
        if record[0]["LinkSetDb"]:
            nucleotide_id = record[0]["LinkSetDb"][0]["Link"][0]["Id"]
            
            # Fetch the nucleotide record
            handle = Entrez.efetch(db="nucleotide", id=nucleotide_id, rettype="gb", retmode="text")
            record = SeqIO.read(handle, "genbank")
            handle.close()
            # return record
            # Extract CDS sequence from the nucleotide record
            for feature in record.features:
                if feature.type == "CDS":
                    try:
                        cand_aa_seq = feature.translate(record.seq)
                    except:
                        try:
                            cand_aa_seq = feature.translate(record.seq.reverse_complement())
                        except:
                            pass

                    if (cand_aa_seq == aa_seq):
                        return str(feature.extract(record.seq))
        # return None  # Return None if no linked nucleotide or CDS found
    except Exception as e:
        print(f"Error retrieving DNA for {protein_id}: {e}")
        return None

#%%
def parallel_fetch_dna_sequences(df, max_workers=6):
    # Initialize the 'dna_seq' column if it doesn't exist
    if 'dna_seq' not in df.columns:
        df['dna_seq'] = None

    # Prepare list of tasks as tuples (protein_id, aa_seq)
    tasks = list(zip(df['id'], df['sequence']))

    # Function to wrap the DNA fetching process
    def fetch_dna(task):
        protein_id, aa_seq = task
        return get_dna_sequence_from_protein(protein_id, aa_seq)

    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and map them to their respective indices
        futures = {executor.submit(fetch_dna, task): idx for idx, task in enumerate(tasks)}

        # Iterate over the completed futures with a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Fetching DNA sequences"):
            idx = futures[future]
            try:
                dna_seq = future.result()
                df.at[idx, 'dna_seq'] = dna_seq
            except Exception as e:
                print(f'Failed to get sequence for id {df.at[idx, "id"]}: {e}')
                df.at[idx, 'dna_seq'] = None

    return df
#%%
non_unprot_df = aa_df[aa_df.db != 'UNIPROT'].reset_index(drop=True)
#%%
new_nup_seq_df = parallel_fetch_dna_sequences(non_unprot_df, max_workers=8)
#%%
new_nup_seq_df[~new_nup_seq_df.dna_seq.isna()]

#%%
missing_up.drop_duplicates('id',inplace=True)
# %%
acquired_uniprot_df.drop_duplicates('id',inplace=True)
# %%
acquired_uniprot_df.set_index('id',inplace=True)
missing_up.set_index('id',inplace=True)
# %%
acquired_uniprot_df.update(missing_up)
# %%
acquired_uniprot_df.reset_index(inplace=True)
#%%
all_df_v2 = pd.concat([acquired_uniprot_df,new_nup_seq_df[~new_nup_seq_df.dna_seq.isna()]])
#%%
all_df_v2.to_csv('all_df_v2.csv')













#%%
len(non_unprot_df)

# %%
new_rf_id = extract_new_refseq_id(gb_record)
# %%
dna_seq = get_dna_sequence_from_protein(new_rf_id)
# %%
dna_seq
# %%
missing_nup
# %%
def fetch_contig_record(contig_id):
    try:
        handle = Entrez.efetch(db="nuccore", id=contig_id, rettype="gb", retmode="text")
        contig_record = SeqIO.read(handle, "genbank")
        handle.close()
        return contig_record
    except Exception as e:
        print(f"Error fetching contig record {contig_id}: {e}")
        return None
# %%
dna_seq.annotations['contig'].split('(')[1].split(':')[0]
# %%
test = fetch_contig_record(dna_seq.annotations['contig'].split('(')[1].split(':')[0])
# %%
len([feat for feat in test.features if feat.type == "CDS"])
# %%
dir(test.features[2])
# %%
test.features[2].extract()
# %%
# %%










#%%
missing_up.drop_duplicates('id',inplace=True)
# %%
acquired_uniprot_df.drop_duplicates('id',inplace=True)
# %%
acquired_uniprot_df.set_index('id',inplace=True)
missing_up.set_index('id',inplace=True)
# %%
acquired_uniprot_df.update(missing_up)
# %%
acquired_uniprot_df.reset_index(inplace=True)
#%%
acquired_uniprot_df
#%%
acquired_uniprot_df = acquired_uniprot_df[acquired_uniprot_df.dna_seq.apply(lambda x:type(x)) == str]
# %%
bad_seqs = acquired_uniprot_df.dna_seq.value_counts()[acquired_uniprot_df.dna_seq.value_counts() > 1].index
acquired_uniprot_df = acquired_uniprot_df[~acquired_uniprot_df.dna_seq.isin(bad_seqs)]
# %%
acquired_uniprot_df.dna_seq.value_counts()[acquired_uniprot_df.dna_seq.value_counts() > 1].index
# %%
acquired_uniprot_df
# %%
acquired_uniprot_df.dna_seq.values[0]
# %%
repeats = acquired_non_uniprot_df[acquired_non_uniprot_df.dna_seq.apply(lambda x:type(x)) == str].dna_seq.value_counts()
repeats[repeats > 1]
# %%
trimmed_non_up = acquired_non_uniprot_df[(~acquired_non_uniprot_df.dna_seq.isin(repeats[repeats > 1].index)) &
                        (acquired_non_uniprot_df.dna_seq.apply(lambda x:type(x) == str))]
# %%
len_ratio = trimmed_non_up.sequence.apply(lambda x:len(x))/trimmed_non_up.dna_seq.apply(lambda x:len(x))
len_ratio[len_ratio > 0.35]
# %%
len_ratio[len_ratio < 0.30]
# %%
(acquired_uniprot_df.sequence.apply(lambda x:len(x))/acquired_uniprot_df.dna_seq.apply(lambda x:len(x))).describe()
# %%
