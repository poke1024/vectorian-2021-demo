# ABSTRACT

## The Vectorian Notebook – An Interactive Environment with Text Similarity Algorithms and Embeddings for the Detection of Intertextuality

*Bernhard Liebl & Manuel Burghardt, Computational Humanities Group, Leipzig University*

The detection of intertextual references in text corpora is a digital humanities topic that has gained a lot of attention in recent years (for instance Bamman & Crane, 2008; Burghardt et al., 2019; Büchler et al., 2013; Forstall et al., 2015; Scheirer et al., 2014). While intertextuality – from a literary studies perspective – describes the phenomenon of one text being present in another text (cf. Genette, 1993), the computational problem at hand is the task of text similarity detection (Bär et al., 2012), and more concretely, semantic similarity detection.

Over the years, there have been various attempts for measuring semantic similarity, some of them knowledge-based (e.g. based on WordNet), others corpus-based, like LDA (Chandrasekaran & Vijay, 2021). The arrival of word embeddings (Mikolov et al., 2013) has changed the field considerably by introducing a new and fast way to tackle the notion of word meaning. On the one hand, word embeddings are building blocks that can be combined with a number of other methods, such as alignments, soft cosine or Word Mover's Distance, to implement some kind of sentence similarity (Manjavacas et al., 2019). On the other hand, the concept of embeddings can be extended to work one the sentence-level as well, which is a conceptually different approach (Wieting et al., 2016).

We introduce the Vectorian as a framework that allows researchers to try out different embedding-based methods for intertextuality detection. In contrast to previous versions of the Vectorian (Liebl & Burghardt, 2020a/b) as a mere web interface with a limited set of static parameters, we now present a clean and completely redesigned API that is showcased in an interactive Jupyter notebook. In this notebook, we first use the Vectorian to build queries where we plug in static word embeddings such as FastText (Mikolov et al., 2018) and GloVe (Pennington et al., 2014). We evaluate the influence of computing similarity through alignments such as Waterman-Smith-Beyer (WSB; Waterman et al., 1976) and two variants of Word Mover’s Distance (WMD; Kusner et al., 2015). We also investigate the performance of state-of-art sentence embeddings like Siamese BERT networks (Reimers & Gurevych, 2019) for the task - both on a document level (as document embeddings) and as contextual token embeddings. Overall, we find that POS tag-weighted WSB with fastText offers highly competitive performance. Readers can upload their own data for performing search queries and try out additional vector space metrics such as p-norms or improved sqrt‐cosine similarity (Sohangir & Wang, 2017).

## References

Bamman, David & Crane, Gregory (2008). The logic and discovery of textual allusion. In Proceedings of the 2008 LREC Workshop on Language Technology for Cultural Heritage Data.

Bär, Daniel, Zesch, Torsten & Gurevych, Iryna (2012). Text reuse detection using a composition of text similarity measures. In Proceedings of COLING 2012, p. 167–184.

Büchler, Marco, Geßner, Annette, Berti, Monica & Eckart, Thomas (2013). Measuring the influence of a work by text re-use. Bulletin of the Institute of Classical Studies. Supplement, p. 63–79.

Burghardt, Manuel, Meyer, Selina, Schmidtbauer, Stephanie & Molz, Johannes (2019). “The Bard meets the Doctor” – Computergestützte Identifikation intertextueller Shakespearebezüge in der Science Fiction-Serie Dr. Who. Book of Abstracts, DHd.

Chandrasekaran, Dhivya & Mago, Vijay (2021). Evolution of Semantic Similarity – A Survey. ACM Computing Surveys (CSUR), 54(2), p. 1-37.
Forstall, Christopher, Coffee, Neil, Buck, Thomas, Roache, Katherine & Jacobson, Sarah (2015). Modeling the scholars: Detecting intertextuality through enhanced word-level n-gram matching. Digital Scholarship in the Humanities, 30(4), p. 503–515.

Genette, Gérard (1993). Palimpseste. Die Literatur auf zweiter Stufe. Suhrkamp.
Kusner, Matt, Sun, Yu, Kolkin, Nicholas & Weinberger, Kilian (2015). From word embeddings to document distances. In International conference on machine learning, p. 957-966.

Liebl, Bernhard & Burghardt, Manuel (2020a). „The Vectorian“ – Eine parametrisierbare Suchmaschine für intertextuelle Referenzen. Book of Abstracts, DHd 2020, Paderborn.

Liebl, Bernhard & Burghardt, Manuel (2020b). “Shakespeare in The Vectorian Age” – An Evaluation of Different Word Embeddings and NLP Parameters for the Detection of Shakespeare Quotes”. Proceedings of the 4th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LateCH), co-located with COLING’2020.

Manjavacas, Enrique, Long, Brian & Kestemont, Mike (2019). On the feasibility of automated detection of allusive text reuse. Proceedings of the 3rd Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature.

Mikolov, Tomas, Chen, Kai, Corrado, Greg & Dean, Jeffrey (2013). Efficient estimation of word representations in vector space. In Proceedings of International Conference on Learning Representations (ICLR 2013). arXiv preprint arXiv:1301.3781.

Mikolov, Tomas, Grave, Edouard, Bojanowski, Piotr, Puhrsch, Christian & Joulin, Armand (2018). Advances in pretraining distributed word representations. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018). arXiv preprint arXiv:1712.09405.

Pennington, Jeffrey, Socher, Richard & Manning, Christopher D. (2014). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), p. 1532-1543.

Reimers, Nils & Gurevych, Iryna (2019). Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP).

Scheirer, Walter, Forstall, Christopher & Coffee, Neil (2014). The sense of a connection: Automatic tracing of intertextuality by meaning. Digital Scholarship in the Humanities, 31(1), p. 204–217.

Sohangir, Sahar & Wang, Dingding (2017). Document Understanding Using Improved Sqrt-Cosine Similarity. In Proceedings of the 2017 IEEE 11th International Conference on Semantic Computing (ICSC), p. 278-279.

Waterman, Michael S., Smith, Temple F. & Beyer, William A. (1976). Some biological sequence metrics. Advances in Mathematics 20(3), p. 367-387.

Wieting, John, Bansal, Mohit, Gimpel, Kevin & Livescu, Karen (2016). Towards universal paraphrastic sentence embeddings. Proceedings of the 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico.

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
|        +-- doc_embeddings.*: precompute Sentence-BERT embeddings for parts of gold.json
|    +-- raw_data
|    |   +-- gold.json: gold standard data for Shakespeare text reuse as JSON 
+--  environment.yml: Python dependencies needed to run the notebook in a conda environment
+--  installation.md: additional documentation how to run this notebook locally or via Binder
+--  miscellaneous: various images used in the notebook
|    +-- output: static images for interactive elements, named by cell number
+--  publication.ipynb: the actual publication as notebook
+-- README.MD: this file
```

# Dependendies

see `environment.yml`

# Authors

Bernhard Liebl, Computational Humanities Group, Leipzig University
liebl@informatik.uni-leipzig.de

Manuel Burghardt, Computational Humanities Group, Leipzig University
burghardt@informatik.uni-leipzig.de
OCRID: 0000-0003-1354-9089
