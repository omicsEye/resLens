import argparse
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.nn.functional import softmax
from tqdm import tqdm

def load_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(f"omicseye/seqsight_4096_512_89M-at-base",trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=config, trust_remote_code=True
    ).to(device)
    model.eval()
    return tokenizer, model, config

def batch_predict(model, tokenizer, sequences, device, batch_size):
    all_preds = []
    all_probs = []
    maxlen = tokenizer.model_max_length
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=maxlen,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    return all_preds, all_probs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary_model_path",    required=True, default='omicseye/resLens_lr_binary')
    p.add_argument("--multiclass_model_path",required=True, default='omicseye/resLens_lr_multiclass')
    p.add_argument("--test_csv",             required=True)
    p.add_argument("--output_csv",           required=True)
    p.add_argument("--batch_size",     type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading binary model...")
    bin_tok, bin_model, _ = load_model_and_tokenizer(args.binary_model_path, device)

    print("Loading multiclass model...")
    multi_tok, multi_model, multi_config = load_model_and_tokenizer(
        args.multiclass_model_path, device
    )

    print("Reading test set...")
    df = pd.read_csv(args.test_csv)

    # Raw sequences
    seqs = df["Sequence"].astype(str).tolist()

    # 1) Binary pass
    print("Running binary predictions...")
    bin_preds, bin_probs = batch_predict(
        bin_model, bin_tok, seqs, device, args.batch_size
    )
    df["binary_pred"] = bin_preds
    df["binary_probs"] = bin_probs

    # 2) Multiclass on those flagged as ARG (binary_pred == 1)
    arg_idx = df.index[df["binary_pred"] == 0].tolist()
    df["arg_pred"] = None
    df["arg_probs"] = None

    if arg_idx:
        arg_seqs = [seqs[i] for i in arg_idx]
        print(f"Running multiclass on {len(arg_seqs)} ARG sequences...")
        multi_preds, multi_probs = batch_predict(
            multi_model, multi_tok, arg_seqs, device, args.batch_size
        )

        # Map numeric IDs → label names, if available
        id2label = getattr(multi_config, "id2label", None) or {}
        # keys might be strings in the checkpoint
        id2label = {int(k): v for k, v in id2label.items()}

        mapped = [ id2label.get(pred, str(pred)) for pred in multi_preds ]
        # convert to Series so that Pandas knows "mapped[i]" goes to df.loc[arg_idx[i]]
        df.loc[arg_idx, "arg_pred"]  = pd.Series(mapped,     index=arg_idx)
        df.loc[arg_idx, "arg_probs"] = pd.Series(multi_probs, index=arg_idx)
    
    # Select and write out
    out_cols = [
        "Sequence",
        "Class",
        "EHS_Naming_Index",
        "db",
        "binary_pred",
        "binary_probs",
        "arg_pred",
        "arg_probs",
    ]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, columns=out_cols, index=False)
    print(f"Done — predictions saved to {args.output_csv}")

if __name__ == "__main__":
    main()
