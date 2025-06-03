# resLens: A genomic language model to enhance antibiotic resistance gene detection

This repo contains the training and inference code for resLens, a family of genomic language models that detect and classify genes that confer antibiotic resistance (ARGs). It leverages language models' contextual understanding of gene function to identify ARGs in a way that is less database dependent than current alignment methods and capable of identifying potential novel ARGs for further investigation.

The scripts directory contains python files to train and evaluate the performance of resLens models and can be adapted to perform inference on other DNA sequence data. It additionally contains the code used for the novel ARG analysis and whole genome sequence analysis performed in the paper.

The example directory contains a Jupyter notebook to perform inference new DNA data, both on a mixed ARG/non-ARG dataset and a purely ARG dataset.

The fine-tuned resLens models, train and test data, and genome IDs and phenotypes for the WGS data can be found at [our HuggingFace repo](https://huggingface.co/collections/omicseye/reslens-6834aae78d6a59c46d156744).

---
**Citation:**  

---