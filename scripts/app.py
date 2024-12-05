import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import plotly.express as px 

# Sequence splitting function
def split_sequence(sequence, max_len=1024, overlap=512):
    chunks = []
    for i in range(0, len(sequence), max_len - overlap):
        chunk = sequence[i:i + max_len]
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, ignore_mismatched_sizes=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def predict_chunk(model, tokenizer, chunk):
    tokens = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
        return outputs.logits

def nucArg_app():
    # Class mappings
    long_read_classes = {
        0: 'aminoglycoside', 1: 'bacitracin', 2: 'beta_lactam', 3: 'chloramphenicol',
        4: 'fosfomycin', 5: 'fosmidomycin', 6: 'fusidic_acid', 7: 'glycopeptide',
        8: 'kasugamycin', 9: 'macrolide-lincosamide-streptogramin', 10: 'multidrug',
        11: 'mupirocin', 12: 'non_resistant', 13: 'peptide', 14: 'polymyxin',
        15: 'qa_compound', 16: 'quinolone', 17: 'rifampin', 18: 'sulfonamide',
        19: 'tetracenomycin', 20: 'tetracycline', 21: 'trimethoprim', 22: 'tunicamycin'
    }
    short_read_classes = {
        0: 'aminoglycoside', 1: 'bacitracin', 2: 'beta_lactam', 3: 'chloramphenicol',
        4: 'fosfomycin', 5: 'fosmidomycin', 6: 'glycopeptide', 7: 'macrolide-lincosamide-streptogramin',
        8: 'multidrug', 9: 'mupirocin', 10: 'polymyxin', 11: 'quinolone',
        12: 'sulfonamide', 13: 'tetracycline', 14: 'trimethoprim'
    }

    # Streamlit UI
    st.title("Detecting Antimicrobial Resistance Genes")
    # st.write("This app predicts antibiotic resistance based on DNA sequences.")

    # Input sequence
    sequence = st.text_area("Enter a DNA sequence:", height=200)

    # Initialize models
    model_long, tokenizer_long = load_model_and_tokenizer("vedantM/NucArg_LongRead")
    model_short, tokenizer_short = load_model_and_tokenizer("vedantM/NucArg_ShortRead")

    if sequence:
        if len(sequence) <= 128:
            st.write("Using Short Reads Model.")
            chunks = [sequence]  # No splitting needed
            model, tokenizer, class_mapping = model_short, tokenizer_short, short_read_classes
        else:
            st.write("Using Long Reads Model.")
            chunks = split_sequence(sequence)
            model, tokenizer, class_mapping = model_long, tokenizer_long, long_read_classes

        # Predict for all chunks and aggregate logits
        all_logits = []
        with st.spinner("Predicting..."):
            for chunk in chunks:
                try:
                    logits = predict_chunk(model, tokenizer, chunk)
                    all_logits.append(logits)
                except Exception as e:
                    st.error(f"Error processing chunk: {e}")
                    return

        # Aggregate logits
        aggregated_logits = torch.mean(torch.stack(all_logits), dim=0)
        probabilities = torch.softmax(aggregated_logits, dim=-1).tolist()
        predicted_class = torch.argmax(aggregated_logits).item()

        # Display results
        # st.success("Prediction complete!")
        st.write("### Prediction complete!")
        st.success(f"Predicted Class: **{class_mapping[predicted_class]}**")
        st.write("### Class Probabilities")
        type_probabilities = []
        for idx, prob in enumerate(probabilities[0]):
            # Append to the new dataset list
            type_probabilities.append({
                'Type': str(class_mapping[idx]),
                'Probability': float(prob)
            })
        
        type_probabilities = pd.DataFrame(type_probabilities).sort_values(by='Probability')
        # type_probabilities = type_probabilities.set_index('Type')
        tp = type_probabilities.convert_dtypes()

        # st.bar_chart(data=tp, horizontal=True, x='Probability', y='Type')
        # df=px.data.tips()
        fig=px.bar(tp,x='Probability',y='Type', orientation='h')
        st.write(fig)


if __name__ == "__main__":
    nucArg_app()
