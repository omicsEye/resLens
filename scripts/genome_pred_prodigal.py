import os
import argparse
import subprocess
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

def load_model(model_id, model_path, device):
    """
    Loads a transformer model and its tokenizer.
    
    Returns:
      tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained("omicseye/seqsight_4096_512_89M-at-base", trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=config, trust_remote_code=True
    ).to(device)
    model.eval()
    return tokenizer, model

def predict_sequences(sequences, tokenizer, model, device, batch_size=16):
    """
    Predicts classes and probability distributions for a list of sequences in batches.
    
    Returns:
      all_predictions: list of predicted labels
      all_probabilities: list of probability arrays
    """
    all_predictions = []
    all_probabilities = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=tokenizer.model_max_length
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
        all_predictions.extend(preds.cpu().numpy())
        all_probabilities.extend(probs.cpu().numpy())
    return all_predictions, all_probabilities

def run_prodigal(fasta_path, results_dir):
    """
    Runs Prodigal on a FASTA file and returns the path to the predicted gene FASTA file.
    If Prodigal fails (non-zero exit status), returns None.
    """
    base = os.path.basename(fasta_path)
    prefix = os.path.splitext(base)[0]
    gene_output_fna = os.path.join(results_dir, f"{prefix}_prodigal_genes.fna")
    gff_output = os.path.join(results_dir, f"{prefix}_prodigal_genes.gff")
    prodigal_cmd = f"prodigal -i {fasta_path} -d {gene_output_fna} -o {gff_output} -f gff"
    print(f"Running Prodigal: {prodigal_cmd}")
    try:
        subprocess.run(prodigal_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Prodigal failed on {fasta_path} with exit code {e.returncode}. Skipping this file.")
        return None
    return gene_output_fna

def filter_genes(gene_fasta, min_length=400):
    """
    Parses a Prodigal output FASTA file and returns gene records longer than min_length.
    """
    try:
        gene_records = list(SeqIO.parse(gene_fasta, "fasta"))
    except Exception as e:
        print(f"Error parsing {gene_fasta}: {e}")
        return []
    filtered = [gene for gene in gene_records if len(gene.seq) >= min_length]
    return filtered

def process_contigs_file(fasta_file, contigs_dir, results_dir, 
                         binary_tokenizer, binary_model, 
                         arg_tokenizer, arg_model, device, batch_size=16, min_length=400):
    """
    Processes a single contigs FASTA file:
      1. Runs Prodigal.
      2. Filters genes longer than min_length.
      3. Runs binary prediction on all genes.
      4. Runs multiclass ARG prediction on genes labeled 1 by the binary model.
    
    Returns a list of result dictionaries.
    """
    fasta_path = os.path.join(contigs_dir, fasta_file)
    gene_fna = run_prodigal(fasta_path, results_dir)
    filtered_genes = filter_genes(gene_fna, min_length)
    print(f"File {fasta_file}: {len(filtered_genes)} genes longer than {min_length} bp found.")
    if not filtered_genes:
        return []
    
    gene_seqs = [str(gene.seq) for gene in filtered_genes]
    
    # Run binary model on all genes
    binary_preds, binary_probs = predict_sequences(gene_seqs, binary_tokenizer, binary_model, device, batch_size)
    
    # For genes predicted as positive (label == 1), run the ARG multiclass model.
    # Initialize lists for multiclass results with None.
    arg_preds = [None] * len(filtered_genes)
    arg_probs = [None] * len(filtered_genes)
    
    positive_indices = [i for i, pred in enumerate(binary_preds) if pred == 0]
    if positive_indices:
        seqs_for_arg = [gene_seqs[i] for i in positive_indices]
        multiclass_preds, multiclass_probs = predict_sequences(seqs_for_arg, arg_tokenizer, arg_model, device, batch_size)
        for idx, pred, prob in zip(positive_indices, multiclass_preds, multiclass_probs):
            arg_preds[idx] = int(pred)
            arg_probs[idx] = prob.tolist()
    
    results = []
    for gene, bin_pred, bin_prob, m_pred, m_prob in zip(filtered_genes, binary_preds, binary_probs, arg_preds, arg_probs):
        results.append({
            "fasta_file": fasta_file,
            "gene_id": gene.id,
            "gene_description": gene.description,
            "gene_sequence": str(gene.seq),
            "binary_prediction": int(bin_pred),
            "binary_prediction_probabilities": bin_prob.tolist(),
            "arg_prediction": m_pred,
            "arg_prediction_probabilities": m_prob
        })
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: Run Prodigal and apply binary and multiclass ARG models on contigs"
    )
    parser.add_argument('--binary_model_id', type=str, required=True, default='resLens_lr_binary',
                        help="ID of the pre-trained binary model")
    parser.add_argument('--binary_model_path', type=str, required=True, default='omicseye/resLens_lr_binary',
                        help="Path to the binary model")
    parser.add_argument('--arg_model_id', type=str, required=True, default = 'resLens_lr_multiclass',
                        help="ID of the pre-trained multiclass ARG model")
    parser.add_argument('--arg_model_path', type=str, required=True, default='omicseye/resLens_lr_multiclass',
                        help="Path to the multiclass ARG model")
    parser.add_argument('--contigs_dir', type=str, required=True,
                        help="Directory containing contig FASTA files and files_to_run.txt")
    parser.add_argument('--output_csv', type=str, required=True,
                        help="Output CSV file to save the final results")
    parser.add_argument('--results_dir', type=str, default="results",
                        help="Directory to store intermediate results (e.g., Prodigal output)")
    parser.add_argument('--batch_size', type=int, default=12,
                        help="Batch size for model predictions")
    parser.add_argument('--min_length', type=int, default=400,
                        help="Minimum gene length (bp) to include in prediction")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading binary model...")
    binary_tokenizer, binary_model = load_model(
        args.binary_model_id, args.binary_model_path, device
    )
    print("Loading multiclass ARG model...")
    arg_tokenizer, arg_model = load_model(
        args.arg_model_id, args.arg_model_path, device
    )

    # Read the list of FASTA files to process
    fasta_files = [f for f in os.listdir(args.contigs_dir)]
    print(f"Will process {len(fasta_files)} files from {args.contigs_dir}")

    all_results = []
    for n, fasta_file in enumerate(fasta_files, 1):
        print(f"\n[{n}/{len(fasta_files)}] Processing: {fasta_file}")
        file_results = process_contigs_file(
            fasta_file, args.contigs_dir, args.results_dir,
            binary_tokenizer, binary_model,
            arg_tokenizer, arg_model,
            device, args.batch_size, args.min_length
        )
        all_results.extend(file_results)

    # Save aggregated results to CSV.
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(args.output_csv, index=False)
        print(f"\nFinished processing. Results saved to {args.output_csv}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
