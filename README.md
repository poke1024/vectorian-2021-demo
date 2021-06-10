# “Embed, embed! There’s knocking at the gate." - Detecting Intertextuality with the Vectorian Notebook of Embeddings

# Abstract

*Bernhard Liebl & Manuel Burghardt, Computational Humanities Group, Leipzig University*

The detection of intertextual references in text corpora is a digital humanities topic that has gained a lot of attention in recent years (for instance Bamman & Crane, 2008; Burghardt et al., 2019; Büchler et al., 2013; Forstall et al., 2015; Scheirer et al., 2014). While intertextuality – from a literary studies perspective – describes the phenomenon of one text being present in another text (cf. Genette, 1993), the computational problem at hand is the task of text similarity detection (Bär et al., 2012), and more concretely, semantic similarity detection.

Over the years, there have been various attempts for measuring semantic similarity, some of them knowledge-based (e.g. based on WordNet), others corpus-based, like LDA (Chandrasekaran & Vijay, 2021). The arrival of word embeddings (Mikolov et al., 2013) has changed the field considerably by introducing a new and fast way to tackle the notion of word meaning. On the one hand, word embeddings are building blocks that can be combined with a number of other methods, such as alignments, soft cosine or Word Mover's Distance, to implement some kind of sentence similarity (Manjavacas et al., 2019). On the other hand, the concept of embeddings can be extended to work one the sentence-level as well, which is a conceptually different approach (Wieting et al., 2016).

We introduce the Vectorian as a framework that allows researchers to try out different embedding-based methods for intertextuality detection. In contrast to previous versions of the Vectorian (Liebl & Burghardt, 2020a/b) as a mere web interface with a limited set of static parameters, we now present a clean and completely redesigned API that is showcased in an interactive Jupyter notebook. In this notebook, we first use the Vectorian to build queries where we plug in static word embeddings such as FastText (Mikolov et al., 2018) and GloVe (Pennington et al., 2014). We evaluate the influence of computing similarity through alignments such as Waterman-Smith-Beyer (WSB; Waterman et al., 1976) and two variants of Word Mover’s Distance (WMD; Kusner et al., 2015). We also investigate the performance of state-of-art sentence embeddings like Siamese BERT networks (Reimers & Gurevych, 2019) for the task - both on a document level (as document embeddings) and as contextual token embeddings. Overall, we find that POS tag-weighted WSB with fastText offers highly competitive performance. Readers can upload their own data for performing search queries and try out additional vector space metrics such as p-norms or improved sqrt‐cosine similarity (Sohangir & Wang, 2017).

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
|    |   +-- sbert_contextual: precomputed Sentence-BERT contextual token embeddings
|    +-- raw_data
|    |   +-- gold.json: gold standard data for Shakespeare text reuse as JSON 
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

