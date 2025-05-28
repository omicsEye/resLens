import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AutoConfig
import torch
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from datasets import load_dataset, Dataset
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
import random
import os

# Argument parser
parser = argparse.ArgumentParser(description='Fine-tune DNA encoder model with 10-fold cross-validation')
parser.add_argument('--model_id', type=str, help='ID of the pre-trained model', required=True)
parser.add_argument('--train_csv', type=str, required=True, help='Path to the training CSV')
parser.add_argument('--test_csv', type=str, required=True, help='Path to the test CSV')
parser.add_argument('--random_seed', type=int, help='Random seed for reproducibility', required=True)
args = parser.parse_args()

# Set the random seed
random_seed = args.random_seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# Get model and dataset names from arguments
model_id = args.model_id
model_name = model_id.split('/')[-1]
dataset_name = args.train_csv.split('.')[0]

# Load the tokenizer - adjustment needed for specific models
if model_name == 'seqsight_4096_512_89M-at-base-multi':
    tokenizer = AutoTokenizer.from_pretrained(f"omicseye/seqsight_4096_512_89M-at-base",trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(f"{model_id}",trust_remote_code=True)

# Load dataset
train_df = pd.read_csv(args.train_csv)
test_df  = pd.read_csv(args.test_csv)

# Convert Class -> binary 'ARG' vs 'non_ARG'
train_df['Class'] = np.where(train_df.Class == 'non_ARG', 'non_ARG', 'ARG')
test_df ['Class'] = np.where(test_df.Class  == 'non_ARG', 'non_ARG', 'ARG')
train_df = train_df.rename(columns={'Class':'type','Sequence':'dna_seq'})
test_df = test_df.rename(columns={'Class':'type','Sequence':'dna_seq'})

# Create label mapping
label_to_id = {label: idx for idx, label in enumerate(train_df['type'].unique())}
id_to_label = {idx: label for label, idx in label_to_id.items()}
print(id_to_label)

train_df['type'] = train_df['type'].map(label_to_id)
test_df ['type'] = test_df['type'].map(label_to_id)

#Set max sequence length
model_max_length = tokenizer.model_max_length

average_char_per_token = 12  # Conservative number of characters per token
token_max_length = model_max_length * average_char_per_token

def split_and_filter_sequence(sequence, token_max_length, min_length=20):
    splits = [sequence[i:i+token_max_length] for i in range(0, len(sequence), token_max_length)]
    return [split for split in splits if len(split) >= min_length]

# Preprocess dataset to split long sequences and filter out short ones
def preprocess_dataset(df):
    new_data = []
    for _, row in df.iterrows():
        original_splits = split_and_filter_sequence(row['dna_seq'], token_max_length)
        # Append both original splits with the same label.
        for split in original_splits:
            new_data.append({
                'dna_seq': split,
                'type': row['type']
            })
    return pd.DataFrame(new_data)

# Preprocess train and test data
train_data = preprocess_dataset(train_df)
test_data = preprocess_dataset(test_df)

print('processed data')
print(f'test data shape: {test_data.shape}')

# Extract sequences and labels
train_sequences = train_data['dna_seq']
train_labels = train_data['type']

test_sequences = test_data['dna_seq']
test_labels = test_data['type']

# Convert to Dataset format
ds_train = Dataset.from_dict({"data": train_sequences, "labels": train_labels})
ds_test = Dataset.from_dict({"data": test_sequences, "labels": test_labels})

# Tokenization function
def tokenize_function(examples):
    outputs = tokenizer(examples["data"],truncation=True,max_length=model_max_length)
    return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"]
            }

# Tokenize test data
tokenized_datasets_test = ds_test.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)

# Cross-validation and training configuration
kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)
batch_size = 16
results = []

# Training arguments function
def get_training_args(fold):
    return TrainingArguments(
        f"checkpoints/{model_name}-finetuned-{dataset_name}-fold{fold}",
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=64,
        num_train_epochs=10,
        logging_steps=200,
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        label_names=["labels"],
        dataloader_drop_last=False,
        report_to="none",
    )

# Metrics computation function
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    unique_labels = len(np.unique(references))
    average_type = 'binary' if unique_labels == 2 else 'weighted'

    return {
        'accuracy': accuracy_score(references, predictions),
        'f1_score': f1_score(references, predictions, average=average_type),
        'mcc': matthews_corrcoef(references, predictions)
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_mcc = 0  # Track best F1 score across all folds
best_model = None  # Track best model
num_labels = train_df.type.nunique()

# Perform 10-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(train_sequences)):
    print(f"Fold {fold+1}")
    torch.cuda.empty_cache()

    # Load model
    config = AutoConfig.from_pretrained(
       f"{model_id}", num_labels=num_labels, trust_remote_code=True
    )

    # Default model loading
    model = AutoModelForSequenceClassification.from_pretrained(
            f"{model_id}", config=config,ignore_mismatched_sizes=True, trust_remote_code=True
        )   

    # Prepare datasets for this fold
    train_sequences_array = train_sequences.to_numpy()
    train_labels_array = train_labels.to_numpy()

    train_fold_sequences = train_sequences_array[train_idx]
    val_fold_sequences = train_sequences_array[val_idx]
    train_fold_labels = train_labels_array[train_idx]
    val_fold_labels = train_labels_array[val_idx]

    ds_train = Dataset.from_dict({"data": train_fold_sequences, 'labels': train_fold_labels})
    ds_val = Dataset.from_dict({"data": val_fold_sequences, 'labels': val_fold_labels})

    # Tokenize datasets
    tokenized_datasets_train = ds_train.map(tokenize_function, batched=True, remove_columns=["data"])
    tokenized_datasets_val = ds_val.map(tokenize_function, batched=True, remove_columns=["data"])

    # Create training arguments for this fold
    args = get_training_args(fold)

    # Initialize Trainer
    trainer = Trainer(
        model.to(device),
        args,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    train_results = trainer.train()

    # Evaluate on validation set
    eval_results = trainer.evaluate()
    mcc_val = eval_results['eval_mcc']
    if mcc_val > best_mcc:
        best_mcc = mcc_val
        best_model = model

    # Save metrics for validation set
    for metric, score in eval_results.items():
        results.append({
            'model': model_name,
            'dataset': dataset_name,
            'evaluation_set': 'validation',
            'metric': metric,
            'score': score
        })

# Evaluate the best model on the test set
best_trainer = Trainer(
    best_model.to(device),
    args,
    eval_dataset=tokenized_datasets_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

test_results = best_trainer.evaluate()

# Save metrics for test set
for metric, score in test_results.items():
    results.append({
        'model': model_name,
        'dataset': dataset_name,
        'evaluation_set': 'test',
        'metric': metric,
        'score': score
    })

# Create 'results' folder if it doesn't exist
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Save results to a CSV file in the 'results' folder
output_file = os.path.join(output_dir, f"{model_name}_binary_model_results_{dataset_name}.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)

# Save the best model
model_save_path = os.path.join(output_dir, f"{model_name}_binary_best_model")
best_model.save_pretrained(model_save_path)

test_predictions = best_trainer.predict(tokenized_datasets_test)
predicted_labels = np.argmax(test_predictions.predictions, axis=1)
predicted_probabilities = test_predictions.predictions
test_data["predicted_labels"] = [id_to_label[label] for label in predicted_labels]
test_data["predicted_probabilities"] = list(predicted_probabilities)
output_file = os.path.join(output_dir, f"{model_name}_binary_test_{dataset_name}_preds.csv")
test_data.to_csv(output_file, index=False)
