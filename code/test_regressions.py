# copied from the notebook but executable as a single script as
# a simple way to test regressions in the computations.

import sys
sys.path.append("code")  # make "nbutils" and "code" importable

import nbutils, gold, vectorian
import ipywidgets as widgets
from ipywidgets import interact

nbutils.initialize("auto")

gold_data = gold.Data("data/raw_data/gold.json")

from vectorian.embeddings import Zoo

the_embeddings = {}

the_embeddings["glove"] = Zoo.load("glove-6B-50")
the_embeddings["fasttext"] = Zoo.load("fasttext-en-mini")

if nbutils.running_inside_binder():  # use precomputed version of Numberbatch?
    the_embeddings["numberbatch"] = nbutils.download_word2vec_embedding(
        "data/raw_data/numberbatch-19.08-en-pca-50",
        "https://zenodo.org/record/4916056/files/numberbatch-19.08-en-pca-50.zip",
    )
else:
    # The following reduction of full Numberbatch to n=50 only works in envs
    # with enough memory. For Binder etc. use the Zenodo version above.
    the_embeddings["numberbatch"] = Zoo.load("numberbatch-19.08-en").pca(50)

from vectorian.embeddings import StackedEmbedding

the_embeddings["fasttext_numberbatch"] = StackedEmbedding(
    [the_embeddings["fasttext"], the_embeddings["numberbatch"]]
)

nlp = nbutils.make_nlp("en_paraphrase_distilroberta_base_v1")

from vectorian.embeddings import SpacyVectorEmbedding, VectorCache

the_embeddings["sbert"] = SpacyVectorEmbedding(
    nlp, 768, cache=VectorCache("data/processed_data/sbert_contextual", readonly=True)
)

from vectorian.session import LabSession
from vectorian.corpus import Corpus

corpus = Corpus("data/processed_data/corpus", mutable=False)

session = LabSession(
    corpus,
    embeddings=the_embeddings.values())

# the following command loads all contextual embedding vectors into RAM.
# while not necessary, this speeds up some of the ensuing computations.
session.cache_contextual_embeddings()

from vectorian.embeddings import CachedPartitionEncoder, SpanEncoder

# create an encoder that basically calls nlp(t).vector
sbert_encoder = CachedPartitionEncoder(
    SpanEncoder(lambda texts: [nlp(t).vector for t in texts])
)

# compute encodings and/or save cached data
sbert_encoder.try_load("data/processed_data/doc_embeddings")
sbert_encoder.cache(session.documents, session.partition("document"))
sbert_encoder.save("data/processed_data/doc_embeddings")

# extract name of encoder for later use
sbert_encoder_name = nlp.meta["name"]


def make_index_builder(**kwargs):
    return nbutils.InteractiveIndexBuilder(
        session, nlp, partition_encoders={sbert_encoder_name: sbert_encoder}, **kwargs
    )

import collections
import ipywidgets as widgets

# define 4 different search stratgies via make_index_builder 
index_builders = collections.OrderedDict(
    {
        "wsb": make_index_builder(
            strategy="Alignment",
            strategy_options={
                "alignment": vectorian.alignment.LocalAlignment(
                    gap={
                        "s": vectorian.alignment.smooth_gap_cost(5),
                        "t": vectorian.alignment.smooth_gap_cost(5)
                    }
                )
            },
        ),
        "wmd nbow": make_index_builder(
            strategy="Alignment",
            strategy_options={
                "alignment": vectorian.alignment.WordMoversDistance.wmd("nbow")
            },
        ),
        "wmd bow": make_index_builder(
            strategy="Alignment",
            strategy_options={
                "alignment": vectorian.alignment.WordMoversDistance.wmd("bow")
            },
        ),
        "doc embedding": make_index_builder(strategy="Partition Embedding"),
    }
)

# present UI of various options that allows for editing
accordion = widgets.Accordion(children=[x.displayable for x in index_builders.values()])
for i, k in enumerate(index_builders.keys()):
    accordion.set_title(i, k)

nbutils.plot_ndcgs(
    gold_data, dict((k, v.build_index()) for k, v in index_builders.items()),
    save_to="ndcg_results.png"
);
