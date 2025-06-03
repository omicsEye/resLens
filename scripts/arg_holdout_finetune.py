import argparse
import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch.nn.functional as F

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "mcc": matthews_corrcoef(labels, preds)
    }


def load_and_encode(df: pd.DataFrame, tokenizer, max_length: int):
    enc = tokenizer(
        df['dna_seq'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    labels = torch.tensor(df['type'].tolist(), dtype=torch.long)
    return TensorDataset(enc['input_ids'], enc['attention_mask'], labels)


def collate_batch(batch):
    input_ids      = torch.stack([x[0] for x in batch], dim=0)
    attention_mask = torch.stack([x[1] for x in batch], dim=0)
    labels         = torch.stack([x[2] for x in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DNA LLM: split remaining into train/val, test on holdout, extract CLS embeddings"
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--remaining", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Load and prepare dataframes
    df_rem = pd.read_csv(args.remaining).rename(columns={"Sequence": "dna_seq", "Class": "type"})
    df_hld = pd.read_csv(args.holdout).rename(columns={"Sequence": "dna_seq", "Class": "type"})
    # remove non-ARG if present
    df_rem = df_rem[df_rem.type != 'non_ARG']

    # Label mapping based on remaining
    labels = sorted(df_rem['type'].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    df_rem['type'] = df_rem['type'].map(label2id)
    df_hld['type'] = df_hld['type'].map(label2id)
    num_labels = len(labels)

    # Split remaining into train and validation
    train_df, val_df = train_test_split(
        df_rem,
        test_size=0.2,
        random_state=args.seed,
        stratify=df_rem['type']
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_id, num_labels=num_labels, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "omicseye/seqsight_4096_512_89M-at-base", trust_remote_code=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Encode datasets
    max_len = getattr(tokenizer, 'model_max_length', 512)
    train_ds = load_and_encode(train_df, tokenizer, max_len)
    val_ds   = load_and_encode(val_df, tokenizer, max_len)
    test_ds  = load_and_encode(df_hld, tokenizer, max_len)

    # Training arguments
    train_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=args.seed,
        dataloader_num_workers=0
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=collate_batch
    )

    # Train
    trainer.train()

    # Evaluate on validation
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    pd.DataFrame([val_metrics]).to_csv(
        os.path.join(args.output_dir, "val_metrics.csv"), index=False
    )

    # Evaluate on holdout (test) and save metrics
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    pd.DataFrame([test_metrics]).to_csv(
        os.path.join(args.output_dir, "test_metrics.csv"), index=False
    )

    # Predict on holdout and save
    preds_out = trainer.predict(test_ds)
    logits = preds_out.predictions
    preds = np.argmax(logits, axis=1)
    probs = F.softmax(torch.tensor(logits), dim=1).cpu().numpy()
    df_preds = df_hld.copy()
    df_preds['true_label'] = df_preds['type'].map(lambda x: id2label[x])
    df_preds['pred_label'] = [id2label[i] for i in preds]
    df_preds['pred_probabilities'] = probs.tolist()
    df_preds.to_csv(
        os.path.join(args.output_dir, "test_predictions.csv"), index=False
    )

    # Extract CLS embeddings for all data
    df_all = pd.concat([train_df, val_df, df_hld], ignore_index=True)
    enc_all = tokenizer(
        df_all['dna_seq'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )
    all_ds = TensorDataset(enc_all['input_ids'], enc_all['attention_mask'])
    loader = DataLoader(all_ds, batch_size=32)

    model.eval()
    embeddings = []
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            cls_emb = out.hidden_states[-1][:, 0, :].cpu()
            embeddings.append(cls_emb)
    all_embeddings = torch.cat(embeddings, dim=0)

    # Save embeddings
    torch.save(all_embeddings, os.path.join(args.output_dir, "all_cls_embeddings.pt"))
    np.save(os.path.join(args.output_dir, "all_cls_embeddings.npy"), all_embeddings.numpy())
    print("Saved embeddings, predictions, and metrics to", args.output_dir)

if __name__ == "__main__":
    main()
