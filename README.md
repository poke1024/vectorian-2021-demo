# “Embed, embed! There’s knocking at the gate." - Detecting Intertextuality with the Vectorian Notebook of Embeddings

# Abstract

*Bernhard Liebl & Manuel Burghardt, Computational Humanities Group, Leipzig University*

The detection of intertextual references in text corpora is a digital humanities topic that has gained a lot of attention in recent years. While intertextuality – from a literary studies perspective – describes the phenomenon of one text being present in another text, the computational problem at hand is the task of text similarity detection, and more concretely, semantic similarity detection. In this notebook, we introduce the Vectorian as a framework to build queries through word embeddings such as fastText and GloVe. We evaluate the influence of computing document similarity through alignments such as Waterman-Smith-Beyer and two variants of Word Mover’s Distance. We also investigate the performance of state-of-art sentence embeddings like Siamese BERT networks for the task - both as document embeddings and as contextual token embeddings. Overall, we find that Waterman-Smith-Beyer with fastText and token similarities weighted by Part-of-speech-tags offers highly competitive performance. The notebook can also be used to upload new data for performing custom search queries.

# Components

```
.
+--  code
|    gold.py: utility functions to read in gold.json data
|    nbbutils.py: various helper code to run publication.ipynb
|    prepare_corpus.ipynb: notebook to recreate the contents of data/processed_data/corpus from gold.json
+--  data
|    +-- processed_data
|    |   +-- corpus: preprocessed Vectorian document data for parts of gold.json (e.g. tokenization)
|    |   +-- doc_embeddings.*: precomputed Sentence-BERT document embeddings for parts of gold.json
|    |   +-- sbert_contextual: precomputed Sentence-BERT contextual token embeddings for pattern phrases
|    +-- raw_data
|    |   +-- gold.json: gold standard data for Shakespeare text reuse as JSON 
|    |   +-- sentence_transformers: will contain S-BERT model (downloaded in the notebook)
|    |   +-- vectorian_cache: will contain word embedding data (downloaded in the notebook)
+--  environment.yml: Python dependencies needed to run the notebook in a conda environment
+--  installation.md: additional documentation how to run this notebook locally or via Binder
+--  miscellaneous: various images used in the notebook
|    +-- output: static images for interactive elements, named by cell number
+--  publication.ipynb: the actual publication as notebook
+-- README.MD: this file
```

# Configuration

You need a Jupyter environment with Python >= 3.7 and various Python packages such as spaCy (for
parsing text) and vectorian (for searching text). We recommend creating a `conda` environment
through the `environment.yml` in this repository (which contains all needed dependencies). To
run the notebook locally in this way, do this:

```
cd /path/to/vectorian/repository
conda env create -f environment.yml
conda activate vectorian-jupyter
jupyter notebook publication.iypnb
```

# Authors

name: Bernhard Liebl
orcid: 0000-0002-8593-400X
institution: Computational Humanities Group, Leipzig University
e-mail: liebl@informatik.uni-leipzig.de
address: Augustusplatz 10, 04109 Leipzig

name: Manuel Burghardt
orcid: 0000-0003-1354-9089
institution: Computational Humanities Group, Leipzig University
e-mail: burghardt@informatik.uni-leipzig.de
address: Augustusplatz 10, 04109 Leipzig

# References

Burghardt, Manuel, Meyer, Selina, Schmidtbauer, Stephanie & Molz, Johannes (2019). “The Bard meets the Doctor” – Computergestützte Identifikation intertextueller Shakespearebezüge in der Science Fiction-Serie Dr. Who. Book of Abstracts, DHd.

Liebl, Bernhard & Burghardt, Manuel (2020a). „The Vectorian“ – Eine parametrisierbare Suchmaschine für intertextuelle Referenzen. Book of Abstracts, DHd 2020, Paderborn.

Liebl, Bernhard & Burghardt, Manuel (2020b). “Shakespeare in The Vectorian Age” – An Evaluation of Different Word Embeddings and NLP Parameters for the Detection of Shakespeare Quotes”. Proceedings of the 4th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LateCH), co-located with COLING’2020.

