#%%
import pandas as pd
from Bio import Entrez, SeqIO
from collections import defaultdict
import math
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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

df = fasta_to_dataframe("../data/database/v1/features.fasta")

# %%
def fetch_uniprot_entry(uniprot_acc):
    """
    Fetches the UniProt entry details for a given UniProt Accession ID.

    Parameters:
    - uniprot_acc (str): The UniProt Accession ID (e.g., 'A0A3B8X2A7').

    Returns:
    - dict: Parsed JSON data of the UniProt entry.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_acc}.json"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        # print(f"Failed to fetch data for {uniprot_acc}: Status Code {response.status_code}")
        return f"Failed to fetch data for {uniprot_acc}: Status Code {response.status_code}"

# %%
def extract_genbank_ids(entry_data):
    """
    Extracts GenBank accession numbers from UniProt entry data.

    Parameters:
    - entry_data (dict): Parsed JSON data of the UniProt entry.

    Returns:
    - list of str: List of GenBank accession numbers.
    """
    genbank_accessions = []
    xrefs = entry_data.get('uniProtKBCrossReferences', [])

    for xref in xrefs:
        database = xref.get('database')
        accession = xref.get('id')
        if database in ['RefSeq', 'GenBank', 'EMBL']:
            if accession:
                genbank_accessions.append(accession)

    return genbank_accessions

# %%
def fetch_cds_sequence(genbank_id, aa_seq):
    """
    Fetches the CDS nucleotide sequence from a GenBank record that translates to the given amino acid sequence.
    
    Parameters:
    - genbank_id (str): The GenBank accession number (e.g., 'FUZR01000002').
    - aa_seq (str): The amino acid sequence of the protein of interest.
    
    Returns:
    - str or None: The nucleotide CDS sequence if a matching CDS is found, else None.
    """
    try:
        # Fetch the GenBank record with features
        handle = Entrez.efetch(db="nucleotide", id=genbank_id[0], rettype="gb", retmode="text")
        record = SeqIO.read(handle, "gb")  # Corrected line without 'alphabet' argument
        handle.close()
    except Exception as e:
        # print(f"[Error] Fetching GenBank record {genbank_id}: {e}")
        return (f"[Error] Fetching GenBank record {genbank_id}: {e}")
    
    # Iterate through CDS features to find a matching translation
    for feature in record.features:
        if feature.type == "CDS":
            try:
                cand_seq = feature.qualifiers['translation'][0]

                if str(cand_seq) == aa_seq:
                    cds_seq = feature.extract(record.seq)
                    if feature.location.strand == -1:
                        cds_seq = cds_seq.reverse_complement()
                    return str(cds_seq)
            except Exception as e:
                # print(f"[Error] Processing CDS feature in {genbank_id}: {e}")
                continue
    
    # If no matching CDS is found
    # print(f"No matching CDS found for the provided amino acid sequence in GenBank accession {genbank_id}.")
    return f"No matching CDS found for the provided amino acid sequence in GenBank accession {genbank_id}."

#%%
def all_seq(uniprot_id,aa_seq):
    try:
        upi_entry = fetch_uniprot_entry(uniprot_id)
        gb_id = extract_genbank_ids(upi_entry)
        cds_seq = fetch_cds_sequence(gb_id, aa_seq)
        return(cds_seq)
    except:
        return(None)

def main_parallel(uniprot_ids, aa_seqs, max_workers=10):
    """
    Fetches CDS sequences for a list of UniProt IDs in parallel.
    
    Parameters:
    - uniprot_ids (list of str): List of UniProt Accession IDs.
    - aa_seqs (list of str): Corresponding amino acid sequences.
    - max_workers (int): Maximum number of threads to use.
    
    Returns:
    - list: List of CDS sequences or error messages.
    """
    cds_seqs = [None] * len(uniprot_ids)  # Pre-allocate list for results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_index = {
            executor.submit(all_seq, uniprot_id, aa_seqs[i]): i
            for i, uniprot_id in enumerate(uniprot_ids)
        }

        # Iterate over the completed futures as they finish
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                cds_seqs[idx] = result
                if idx % 500 == 0:
                    logging.info(f"{idx} records processed")
            except Exception as e:
                logging.error(f"Error processing index {idx} (UniProt ID: {uniprot_ids[idx]}): {e}")
                cds_seqs[idx] = None

    return cds_seqs
# %%
uniprot_ids = df[df.db == 'UNIPROT'].id.values
aa_seqs = df[df.db == 'UNIPROT'].sequence.values
# %%
cds_seqs = main_parallel(uniprot_ids, aa_seqs, max_workers=8)

#%%
#%%
len(cds_seqs)
#%%
allowed_chars = set('ATCG')
filtered = [
        seq for seq in cds_seqs
        if seq
]

# filtered = [
#         seq for seq in cds_seqs
#         if all(char.upper() in allowed_chars for char in seq) > 0
#     ]
len(filtered)
#%%
import numpy as np
seq_arr = pd.Series(cds_seqs)
#%%
filtered = seq_arr[(seq_arr.apply(lambda x:type(x)) == str)]
len(filtered[~filtered.apply(lambda x:x[0]).isin(['[','N'])])
#%%
uniprot_df = df[df.db == 'UNIPROT']
uniprot_df['dna_seq'] = cds_seqs
#%%
uniprot_df.to_csv('uniprot_dna_seq.csv')

#%%
def get_proper_ids(uniprot_ids):
    url = "https://rest.uniprot.org/uniparc/search"
    proper_ids = []
    for n,id in enumerate(uniprot_ids):
        if n%500 == 0:
            print(n)
        try:
            params = {
                "query": id,
                "format": "json"
                # "size": batch_size  # Adjust based on expected number of entries
            }
            response = requests.get(url, params=params)
            proper_ids.append(response.json()['results'][0]['uniParcCrossReferences'][0]['id'])
        except:
            proper_ids.append(f"No ID found for {id}")
    return proper_ids

def get_refseq_ids(uniprot_ids):
    url = "https://rest.uniprot.org/uniparc/search"
    proper_ids = []
    for n,id in enumerate(uniprot_ids):
        try:
            params = {
                "query": id,
                "format": "json"
                # "size": batch_size  # Adjust based on expected number of entries
            }
            response = requests.get(url, params=params)
            potential_ids = []
            for ref in response.json()['results'][0]['uniParcCrossReferences']:
                if ref['database'] == 'RefSeq':
                    potential_ids.append(ref['id'])
            if len(potential_ids) == 0:
                potential_ids.append(f"no refseq id found for {id}")

            proper_ids.append(potential_ids[0])
        except:
            proper_ids.append(f"No ID found for {id}")
    return proper_ids

#%%
test_ids = get_proper_ids(uniprot_ids[:100])

#%%
len(test_ids)
#%%
len([id for id in test_ids if id[0] != '[' & id[0] != 'N'])

# %%
uniprot_df = df[df.db == 'UNIPROT']
uniprot_df['proper_id'] = test_ids

#%%
cds_seqs = []
for i,uniprot_id in enumerate(test_ids):
    if i%10 == 0:
        print(i,"records processed")
    try:
        upi_entry = fetch_uniprot_entry(uniprot_id)
        gb_id = extract_genbank_ids(upi_entry)
        if len(gb_id) == 0:
            print(f"no gb_ids found for {uniprot_id}")
        cds_seq = fetch_cds_sequence(gb_id, aa_seqs[i])
        cds_seqs.append(cds_seq)
    except:
        cds_seqs.append(f"No CDS found for {uniprot_id}.")
# %%
cds_seqs
# %%
len([seq for seq in cds_seqs if seq != "[Error] Fetching GenBank record []: No records found in handle"])

#%%
len([seq for seq in cds_seqs if seq != "[Error] Fetching GenBank record []: No records found in handle"])

# %%
uniprot_df.proper_id.values[-8]
# %%
uniprot_df.id.values[-8]

# %%
test_resp = get_proper_ids(uniprot_df.id.values[3])
# %%
test_resp
# %%
url = "https://rest.uniprot.org/uniparc/search"
id = "A0A0C1BQ13"
params = {
    "query": id,
    "format": "json"
    # "size": batch_size  # Adjust based on expected number of entries
}
response = requests.get(url, params=params)
# %%
response.json()['results'][0]['uniParcCrossReferences']
# %%
test_refseq_ids = get_refseq_ids(uniprot_ids[:100])
# %%
len(test_refseq_ids)

# %%
len([id for id in test_refseq_ids if id[:2] != "no"])
# %%
test_refseq_ids
# %%
def get_nucleotide_accession_ids(refseq_protein_id):
    try:
        # Use elink to find linked nucleotide records
        handle = Entrez.elink(dbfrom="protein", db="nucleotide", id=refseq_protein_id)
        data = Entrez.read(handle)
        handle.close()

        nucleotide_refseq_ids = []
        print(data)
        if data and data[0].get("LinkSetDb"):
            for linksetdb in data[0]["LinkSetDb"]:
                for link in linksetdb["Link"]:
                    print(link)
                    nucleotide_uid = link["Id"]
                    return nucleotide_uid
        else:
            print(f"  [Warning] No nucleotide links found for {refseq_protein_id}.")

    except Exception as e:
        print(f"  [Error] Failed to retrieve nucleotide RefSeq IDs for {refseq_protein_id}: {e}")
        return []
# %%
test_ncl_id = get_nucleotide_accession_ids(test_refseq_ids[1])
# %%
test_ncl_id
# %%
def fetch_genbank_record(refseq_protein_id):
    try:
        handle = Entrez.efetch(db="protein", id=refseq_protein_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        print(f"[Info] Successfully fetched GenBank record for {refseq_protein_id}.")
        print(record)
        return record
    except Exception as e:
        print(f"[Error] Failed to fetch GenBank record for {refseq_protein_id}: {e}")
        return None
    
def extract_cds_sequences(genbank_record, refseq_protein_id):
    cds_sequences = []
    for feature in genbank_record.features:
        if feature.type == "CDS":
            # Attempt to get the protein_id from qualifiers
            protein_ids = feature.qualifiers.get('protein_id', [])
            if not protein_ids:
                # Fallback: Check 'product' or other qualifiers if needed
                continue
            protein_id = protein_ids[0]
            if protein_id != refseq_protein_id:
                continue  # Skip CDS features not matching the target protein

            # Get CDS ID or generate one
            cds_id = feature.qualifiers.get('locus_tag', [''])[0] or feature.qualifiers.get('gene', [''])[0]
            if not cds_id:
                # Generate a unique identifier if none exists
                cds_id = f"CDS_{len(cds_sequences)+1}"

            # Extract the nucleotide sequence of the CDS
            try:
                cds_seq = feature.extract(genbank_record.seq)
                # Handle reverse complement if on the negative strand
                if feature.location.strand == -1:
                    cds_seq = cds_seq.reverse_complement()
                cds_sequences.append((cds_id, str(cds_seq)))
                print(f"  [CDS] Extracted CDS ID: {cds_id}")
            except Exception as e:
                print(f"  [Error] Failed to extract CDS sequence: {e}")
                continue
    return cds_sequences
# %%
test_gb_record = fetch_genbank_record(test_refseq_ids[2])
test_cds_seq = extract_cds_sequences(test_gb_record, test_refseq_ids[2])
# %%
test_cds_seq
# %%













#%%
non_unifprot_df = df[df.db != 'UNIPROT']
# %%
def get_dna_sequence_from_protein(protein_id):
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
            
            # Extract CDS sequence from the nucleotide record
            for feature in record.features:
                if feature.type == "CDS":
                    cds_sequence = feature.extract(record.seq)
                    return str(cds_sequence)
        return None  # Return None if no linked nucleotide or CDS found
    except Exception as e:
        print(f"Error retrieving DNA for {protein_id}: {e}")
        return None
    
def parallel_retrieve_cds_sequences(df, max_workers=1):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_protein_id = {executor.submit(get_dna_sequence_from_protein, protein_id): protein_id for protein_id in df['id']}
        results = []
        
        # Collect results as they complete
        for i,future in enumerate(as_completed(future_to_protein_id)):
            if i%100 == 0:
                print(i)
            protein_id = future_to_protein_id[future]
            try:
                cds_sequence = future.result()
                results.append({'Protein_Accession_ID': protein_id, 'CDS_Sequence': cds_sequence})
            except Exception as e:
                print(f"Error processing {protein_id}: {e}")
                results.append({'Protein_Accession_ID': protein_id, 'CDS_Sequence': None})
                
    return pd.DataFrame(results)
# %%
cds_df = parallel_retrieve_cds_sequences(non_unifprot_df)
# %%
cds_df[cds_df.CDS_Sequence.apply(lambda x:type(x)) == str]
# %%
non_unifprot_df.
# %%
uniprot_df = pd.read_csv('uniprot_dna_seq.csv')
# %%
uniprot_df
# %%
non_unifprot_df = pd.merge(non_unifprot_df,cds_df,left_on='id',right_on='Protein_Accession_ID')
#%%
non_unifprot_df.to_csv('non_uniprot_dna_seq.csv')
# %%
non_unifprot_df.columns = ['id', 'db', 'type', 'sequence', 'Protein_Accession_ID', 'dna_seq']
# %%
non_unifprot_df = non_unifprot_df[['id', 'db', 'type', 'sequence', 'dna_seq']]
uniprot_df = uniprot_df[['id', 'db', 'type', 'sequence', 'dna_seq']]

# %%
all_df = pd.concat([non_unifprot_df,uniprot_df])
# %%
all_df = all_df[all_df.dna_seq.apply(lambda x:type(x)) == str]
all_df = all_df[~all_df.dna_seq.apply(lambda x:x[0]).isin(['[','N'])]
# %%
all_df.to_csv('all_df_v1.csv')
# %%
