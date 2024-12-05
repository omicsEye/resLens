# Deep Learning for Antibiotic Resistance Gene Detection

PUBH 8885: Computational Biology
Fall 2024
Vedant Mahangade, Matthew Mollerus, Lucia Sanchez

This repo contains our work reproducing the work of [DeepARG](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-018-0401-z) by Arango-Argoty et al., as well as our efforts to expand upon it by finetuning a DNA language model for ARG detection.

The DeepARG directory contains the Jupyter notebooks in which we recreated their work (called 'reproduce_model_type.ipynb' for each type of model) for both short read and long read models and expanded upon it by training models that could use DNA rather than amino acid sequences as inputs. Additionally, it contains notebooks that processed their data and retrieved DNA sequences for their original amino acid sequences and to perform ablation testing through training linear models on their features.

The scripts directory contains notebooks for training the large language model, as well as for testing we performed on it's output and performance.

Overall, we failed to precisely reproduce their results, we believe due to discrepancies between the dataset they described and the one they made publically available, as well as some unclear points in their description of their modelling. The models we reproduced showed the following results:

* Models that took amino acid sequences as inputs generally performed better than equivalent models that took DNA sequences as inputs; however, the latter may be more useful in realworld metagenomic applications.
* Ablation testing with linear models shows that for alignment based methods, such as that implemented in the DeepARG paper, the complexity of deep learning is likely unjustified, as linear models performed similarly or better.
* Finetuned language models outperformed the relevant alignment-based models, indicating that their contextual understanding of sequence data may produce superior results.

More details of our implementation and results can be found in the notebooks and in the slides in this directory.