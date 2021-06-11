import spacy
import spacy_sentence_bert
import string
import collections
import numpy as np
import math
import os
import sys
import enum
import sklearn.metrics
import itertools
import functools
import ipywidgets as widgets
import jp_proxy_widget
import requests
import markdown
import openTSNE
import openTSNE.callbacks
import networkx as nx
import IPython.core.display
import xml.etree.ElementTree as ET
import zipfile
import io
import requests

import bokeh.plotting
import bokeh.models
import bokeh.transform
import bokeh.palettes
import bokeh.layouts
import bokeh.io

from functools import partial
from cached_property import cached_property
from IPython.core.display import HTML, display
from bs4 import BeautifulSoup
from tqdm.autonotebook import tqdm
from pathlib import Path
from contextlib import contextmanager

os.environ["VECTORIAN_CACHE_HOME"] = "data/raw_data/vectorian_cache"
os.environ["GENSIM_DATA_DIR"] = "data/raw_data/gensim_data"

sbert_cache_path = Path("data/raw_data/sentence_transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(sbert_cache_path)

if os.environ.get("VECTORIAN_DEV"):
    os.environ["VECTORIAN_CPP_IMPORT"] = "1"
    vectorian_path = Path("/Users/arbeit/Projects/vectorian-2021")
    sys.path.append(str(vectorian_path))
    import vectorian

from vectorian.embeddings import Word2VecVectors, TokenEmbeddingAggregator, prepare_docs
from vectorian.embeddings import CachedPartitionEncoder
from vectorian.index import DummyIndex
from vectorian.metrics import TokenSimilarity, CosineSimilarity
from vectorian.interact import PartitionMetricWidget


class DisplayMode(enum.Enum):
    SERVER = (True, False)
    BINDER = (False, False)
    EXPORT = (True, True)
    
    def __init__(self, bokeh, static):
        self.bokeh = bokeh  # prefer bokeh widgets
        self.static = static  # static or interactive?
        
    @property
    def fully_interactive(self):
        return self.bokeh and not self.static
    

_display_mode = DisplayMode.EXPORT

def default_plot_width():
    if _display_mode.static:
        return 600
    else:
        return 1000

    
def running_inside_binder():
    return os.environ.get("BINDER_SERVICE_HOST") is not None


def initialize(display_mode="auto"):
    global _display_mode
    if display_mode == "auto":
        _display_mode = DisplayMode.BINDER if running_inside_binder() else DisplayMode.SERVER
    else:
        _display_mode = DisplayMode[display_mode.upper()]
    display(HTML(f"""
        <div>Running notebook in <b>{_display_mode.name}</b> mode.</div>"""))
    
    if _display_mode == DisplayMode.SERVER:
        os.environ["BOKEH_ALLOW_WS_ORIGIN"] = ",".join(
            [f"localhost:{port}" for port in range(8888, 8898)])
    
    bokeh.io.output_notebook()
        

def _bokeh_show(root):
    bokeh.io.show(root)
    #bokeh.io.export_png(root, filename="/Users/arbeit/Desktop/bokeh.png", width=1200)
    #bokeh.io.export_svg(root, filename="/Users/arbeit/Desktop/bokeh.svg")


def make_limited_function_warning_widget(action_text):
    more_text = ("Run this notebook on a local Jupyter installation (with Bokeh server support) in order to " +
        f"interactively {action_text}.")
    info_text = ('&#x26a0; This visualization has some limited interactivity due to technical constraints in Binder. ' + 
        f'<a href="#" onclick="alert(\'{more_text}\');">learn more...</a>')
    return widgets.HTML(
        value=f'<b><p style="text-align:center"><font color="#F4D03F">{info_text}</p></b>',
        layout=widgets.Layout(border='solid 1px #F9E79F'))


@contextmanager
def monkey_patch_sentence_transformers_tqdm(desc):
    # patch progress so we know that it's actually sentence_transformers we are seeing,
    # since sentence_transformers does not set "desc" on its tqdm instance :-/
    from functools import partial
    import sentence_transformers.util

    old_tqdm = sentence_transformers.util.tqdm
    sentence_transformers.util.tqdm = partial(tqdm, desc=desc)
    try:
        yield
    finally:
        sentence_transformers.util.tqdm = old_tqdm


# the following function is adapted from:
# https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=f"Downloading {url}",
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def download_word2vec_embedding(name, url):
    data_path = Path(f"{name}.zip")
    if not data_path.exists():
        download(url, data_path)

    with zipfile.ZipFile(data_path, 'r') as zf:
        for zi in zf.infolist():
            if zi.filename[-1] == '/':
                continue

            data = zf.read(zi)

            return Word2VecVectors(
                name,
                io.BytesIO(data),
                binary=True)

    raise ValueError("zip file is empty")


def make_nlp(sbert_model_name):
    # uses 'tagger' from en_core_web_sm
    # we include 'parser' so that Vectorian can detect sentence boundaries

    sbert_model_path = sbert_cache_path / "sbert.net_models_paraphrase-distilroberta-base-v1"
    sbert_model_zip_path = sbert_model_path.parent / (sbert_model_path.name + ".zip")

    if not sbert_model_zip_path.exists():
        download(
            "https://zenodo.org/record/4923260/files/sbert.net_models_paraphrase-distilroberta-base-v1.zip",
            sbert_model_zip_path)
    if not sbert_model_path.is_dir():
        with zipfile.ZipFile(sbert_model_zip_path, "r") as zf:
            zf.extractall(sbert_model_path.parent)

    with monkey_patch_sentence_transformers_tqdm("Downloading Sentence BERT model"):
        nlp = spacy.load('en_core_web_sm', exclude=['ner'])
        nlp.add_pipe('sentence_bert', config={'model_name': sbert_model_name})
        nlp.meta["name"] = "core_web_sm_AND_" + sbert_model_name
        return nlp


def occ_digest(occ, n=80):
    return f"{occ.source.work}: {occ.evidence.context}"[:n] + "..."
                
    
class OccurenceFormatter:
    def __init__(self):
        pass
    
    def format_occurrence(self, occurence):
        text = occurence.evidence.context
        quote = occurence.evidence.phrase
        try:
            quote_parts = [s.strip() for s in quote.split("[...]")]
            
            i0 = 0
            indices = []
            for q in quote_parts:
                i = text.index(q, i0)
                indices.append((i, q))
                i0 = i + len(q)
                
            for i, q in reversed(indices):
                text = ''.join([
                    text[:i],
                    '<span style="font-weight:bold;">',
                    text[i:i + len(q)],
                    '</span>',
                    text[i + len(q):]
                ])
                
            return text
        except:
            return text
            
    
class DocFormatter:
    def __init__(self, gold):
        self._gold = gold
        self._occ_by_id = dict((x.unique_id, x) for x in gold.occurrences)
        
        self._fmt = OccurenceFormatter()
        self._template = string.Template("""
            <div style="margin-left:2em">
                <span style="font-variant:small-caps; font-size: 14pt;">${title}</style>
                <span style="float:right; font-size: 10pt;">query: ${phrase}</span>
                <hr>
                <div style="font-variant:normal; font-size: 10pt;">${text}</div>
            </div>
            """)
       
    def format_occurrence(self, doc):
        return self._fmt.format_occurrence(
            self._occ_by_id[doc.unique_id])
        
    def format_doc(self, doc):
        return self._template.substitute(
            phrase=self._occ_by_id[doc.unique_id].pattern.phrase,
            title=doc.metadata["title"],
            text=self.format_occurrence(doc))


def format_embedding_name(name):
    if name.startswith("https://github.com"):
        return "/".join(name.split("/")[4:])
    else:
        return name

    
def find_index_by_filter(terms, s):
    candidates = []
    for i, x in enumerate(terms):
        if s in x:
            candidates.append((i, x))
    if not candidates:
        raise ValueError(f"did not find '{s}' in {terms}")
    return min(candidates, key=lambda x: len(x[1]))[0]

    
class Browser:
    def __init__(self, gold, initial_phrase=1, initial_context=1, rows=5):
        self._gold = gold
        
        formatter = OccurenceFormatter()

        if isinstance(initial_phrase, str):
            initial_phrase = find_index_by_filter(
                [p.phrase for p in gold.patterns], initial_phrase) + 1

        pattern_select = widgets.Dropdown(
            options=[(p.phrase, i) for i, p in enumerate(gold.patterns)],
            value=initial_phrase - 1,
            #rows=rows,
            description='pattern:',
            layout={'description_width': 'initial', 'width': 'max-content'})
        self._pattern_select = pattern_select

        def query_contexts():
            occurrences = gold.patterns[int(pattern_select.value)].occurrences
            return [(x.source.work, i) for i, x in enumerate(occurrences)]

        if isinstance(initial_context, str):
            initial_context = find_index_by_filter(
                [x[0] for x in query_contexts()], initial_context) + 1

        context_select = widgets.Dropdown(
            options=query_contexts(),
            value=initial_context - 1,
            #rows=rows,
            description='occurs in:',
            layout={'description_width': 'initial', 'width': 'max-content'})
        self._context_select = context_select

        if False and _display_mode.static:
            occurrences = gold.patterns[pattern_select.value].occurrences

            html = f"""
            The phrase <b><span style="background-color:#F0F0E0">{gold.patterns[pattern_select.value].phrase}</span></b> occurs in
            <em>{query_contexts()[context_select.value][0]}</em> as:<br>
            <div style="background-color:#F0F0E0">{formatter.format_occurrence(occurrences[context_select.value])}</div>
            """

            display(IPython.core.display.HTML(html))

        elif _display_mode.static:

            pattern_select = bokeh.models.Select(
                options=[(str(i), p.phrase) for i, p in enumerate(gold.patterns)],
                value=str(initial_phrase - 1),
                title='pattern:',
                width=300)

            context_select = bokeh.models.Select(
                options=[(str(v), k) for k, v in query_contexts()],
                value=str(initial_context - 1),
                title='occurs in:',
                width=300)

            occurrences = gold.patterns[int(pattern_select.value)].occurrences
            works = query_contexts()
            i = int(context_select.value)
                
            context_display = bokeh.models.Div(
                text=formatter.format_occurrence(occurrences[i]), width=550)
                
            root = bokeh.layouts.column(
                bokeh.layouts.row(pattern_select, context_select),
                bokeh.layouts.row(bokeh.models.Div(text="as:", width=50), context_display))
                
            _bokeh_show(root)
                        
        else:
            context_display = widgets.HTML(
                "", description="as:", style={'description_width': 'initial', 'width': 'max'})

            def on_phrase_change(change):
                context_select.unobserve(on_context_change)
                context_select.options = query_contexts()
                context_select.value = 0
                context_select.observe(on_context_change)
                on_context_change(None)

            def on_context_change(change):
                occurrences = gold.patterns[pattern_select.value].occurrences
                works = query_contexts()
                i = context_select.value
                context_display.value = formatter.format_occurrence(occurrences[i]) + f' <span style="float:right;"><i>{occurrences[i].unique_id}</i></span>'

            pattern_select.observe(on_phrase_change)
            context_select.observe(on_context_change)
            on_context_change(None)

            display(widgets.VBox([
                widgets.HBox([pattern_select, context_select]),
                context_display
            ]))
            
    @property
    def occurrence(self):
        occurrences = self._gold.patterns[self._pattern_select.value].occurrences
        return occurrences[self._context_select.value]
    
    
    
class TokenSimPlotterFactory:
    def __init__(self, session, nlp, gold):
        self._session = session
        self._nlp = nlp
        self._gold = gold
    
    def make(self, initial_phrase=1, initial_context=1):
        browser = Browser(self._gold, initial_phrase, initial_context)

        def plot(pivot):
            plot_token_similarity(
                self._session, self._nlp, self._gold, pivot, browser.occurrence, n_figures=3)
   
        return plot

    

class TSNECallback(openTSNE.callbacks.Callback):
    def optimization_about_to_start(self):
        pass

    def __call__(self, iteration, error, embedding):
        pass

    
class DocEmbedderFactory:
    def __init__(self, session, nlp, doc_encoders={}):
        self._session = session
        self._nlp = nlp
        self._doc_encoders = doc_encoders
    
    def create(self, encoder):
        return DocEmbedder(self._session, self._nlp, self._doc_encoders, encoder)

    
class DocEmbedder:
    def __init__(self, session, nlp, doc_encoders={}, encoder=None):
        self._session = session
        self._nlp = nlp
        self._doc_encoders = doc_encoders
        self._callbacks = []

        self._partition = session.partition("document")
        
        Option = collections.namedtuple("Option", ["name", "token_embedding", "doc_encoder"])
        
        options = []
        for k, v in self._session.embeddings.items():
            options.append(Option(format_embedding_name(k) + " [token]", v, None))
        for k, v in doc_encoders.items():
            options.append(Option(format_embedding_name(k) + " [doc]", None, v))
        self._options = options

        default_option = options[0].name
        if encoder is not None:
            default_option = options[find_index_by_filter(
                [x.name for x in options], encoder)].name
        
        agg_options = ["mean", "median", "max", "min"]
        
        if _display_mode.bokeh:
            self._embedding_select = bokeh.models.Select(
                title="",
                value=default_option,
                options=[option.name for option in options],
                width=400)

            self._aggregator = bokeh.models.Select(
                title="",
                value=agg_options[0],
                options=agg_options)

            def embedding_changed_shim(attr, old, new):
                self.embedding_changed()

            def aggregator_changed_shim(attr, old, new):
                self.aggregator_changed()
            
            if not _display_mode.static:
                self._embedding_select.on_change("value", embedding_changed_shim)
            self.embedding_changed()

            if not _display_mode.static:
                self._aggregator.on_change("value", aggregator_changed_shim)
        else:
            self._embedding_select = widgets.Dropdown(
                title="",
                layout={'description_width': 'initial', 'width': 'max-content'},
                value=default_option,
                options=[option.name for option in options])

            self._aggregator = widgets.Dropdown(
                title="",
                layout={'description_width': 'initial', 'width': 'max-content'},
                value=agg_options[0],
                options=agg_options)
            
            self._embedding_select.observe(lambda changed: self.embedding_changed(), names="value")
            self.embedding_changed()

            self._aggregator.observe(lambda changed: self.aggregator_changed(), names="value")
             
    @property
    def disabled(self):
        return self._embedding_select.disabled
                
    @disabled.setter
    def disabled(self, value):
        self._embedding_select.disabled = value
        self._aggregator.disabled = value
                
    def _change_occured(self):
        for cb in self._callbacks:
            cb()

    def embedding_changed(self):
        visible = self.option.token_embedding is not None
        if _display_mode.bokeh:
            self._aggregator.visible = visible
        else:
            self._aggregator.layout.display = 'block' if visible else 'none'
            
        self._change_occured()
            
    def aggregator_changed(self):
        self._change_occured()

    @property
    def session(self):
        return self._session

    @property
    def option(self):
        return self._options[[x.name for x in self._options].index(self._embedding_select.value)]
    
    def on_change(self, callback):
        self._callbacks.append(callback)

    @property
    def widget(self):       
        if _display_mode.bokeh:
            return bokeh.layouts.row(self._embedding_select, self._aggregator)
        else:
            return widgets.HBox([self._embedding_select, self._aggregator])
    
    def display(self):
        if _display_mode.bokeh:
            bokeh.io.show(lambda doc: doc.add_root(self.widget))
        else:
            display(self.widget)
            
    @property
    def encoder(self):
        option = self.option
        if option.doc_encoder is not None:
            return option.doc_encoder
        else:
            agg = getattr(np, self._aggregator.value)
            return CachedPartitionEncoder(
                TokenEmbeddingAggregator(option.token_embedding.factory, agg))
 
    @property
    def partition(self):
        return self._partition
    
    def mk_query(self, text):
        return DummyIndex(self.partition).make_query(text)
    
    def encode(self, docs):
        return self.encoder.encode(
            prepare_docs(docs, self._nlp), self.partition).unmodified
    

class EmbeddingPlotter:    
    def __init__(self, embedder, gold):
        self._embedder = embedder
        self._gold = gold

        self._current_selection = None

        self._id_to_doc = dict((doc.unique_id, doc) for doc in self.session.documents)
        self._doc_formatter = DocFormatter(gold)
        
        DocData = collections.namedtuple("DocData", ["doc", "query", "work"])

        docs = []
        for pattern in self._gold.patterns:
            for occ in pattern.occurrences:
                docs.append(DocData(
                    doc=self._id_to_doc[occ.unique_id],
                    query=pattern.phrase,
                    work=occ.source.work))
        self._docs = docs
         
        self._doc_emb_tooltips = """
            <span style="font-variant:small-caps">@work</span>
            <br>
            <span style="float:left;">"@query" (@similarity%)</span>
            <br>
            <hr>
            @context
            """

        self._tok_emb_tooltips = """
            <span style="font-variant:small-caps">@work</span>
            <br>
            <span style="float:left;">"@query"</span>
            <br>
            <hr>
            @context
            """
    
        tsne_callback = TSNECallback()
    
        self._doc_tsne = openTSNE.TSNE(
            perplexity=30,
            metric="cosine",
            callbacks=tsne_callback,
            n_jobs=2,
            random_state=42)

        self._tok_tsne = openTSNE.TSNE(
            perplexity=50,  # 10
            metric="cosine",
            callbacks=tsne_callback,
            n_jobs=2,
            random_state=42)
        
        self._empty_token_data = {
            'x': [],
            'y': [],
            'query': [],
            'token': [],
            'work': [],
            'context': []
        }
        
        self._tok_plot_state = 0

        self._figures = []
        self._figures_html = []

    @property
    def session(self):
        return self._embedder.session
    
    @property
    def partition(self):
        return self._embedder.partition
    
    def _compute_source_data(self, intruder):
        intruder_doc = self._embedder.mk_query(intruder)
        
        id_to_doc = self._id_to_doc
        query_docs = []
        
        works = []
        phrases = []
        contexts = []

        query_docs.append(intruder_doc)
        works.append("")
        phrases.append(intruder)
        contexts.append("")

        for occ in self._gold.occurrences:
            doc = id_to_doc[occ.unique_id]
            query_docs.append(doc)
            works.append(occ.source.work)
            phrases.append(occ.pattern.phrase)
            contexts.append(self._doc_formatter.format_occurrence(doc))

        data = {
            'work': works,
            'query': phrases,
            'context': contexts,
            'vector': self._embedder.encode(query_docs)
        }
        
        include_intruder = not np.any(np.isnan(data['vector']))
        if include_intruder:
            qs = 1
        else:
            for k in data.keys():
                data[k] = data[k][1:]
            qs = 0

        v = np.array(data['vector'])
        v /= np.linalg.norm(v, axis=1)[:, np.newaxis]

        similarity = [1]
        for x in v[1:]:
            similarity.append(np.dot(v[0], x))
        similarity = np.array(similarity) * 100
        
        X = self._doc_tsne.fit(v)

        return {
            'docs': {
                'x': X[qs:, 0],
                'y': X[qs:, 1],
                'work': data["work"][qs:],
                'query': data["query"][qs:],
                'context': data["context"][qs:],
                'similarity': similarity[qs:]
            },
            'query': {
                'x': X[:qs, 0],
                'y': X[:qs, 1],
                'work': data["work"][:qs],
                'query': data["query"][:qs],
                'context': data["context"][:qs],
                'similarity': similarity[:qs]
            },
            'tokens': self._empty_token_data
        }
    
    @property
    def selection(self):
        return self._current_selection
    
    def _update_figures_html(self):
        if not self._figures:
            return
        
        script, divs = bokeh.embed.components(self._figures)
        for html, div in zip(self._figures_html, divs):
            html.value = div
            
        script_code = []
        for x in BeautifulSoup(script, features="html.parser").findAll("script"):
            script_code.append(x.string)
        self._pw.js_init("\n".join(script_code))
                
    def mk_plot(self, bokeh_doc, selection=[], locator=None, plot_width=1200):
        has_tok_emb = self._embedder.option.token_embedding is not None
        
        if _display_mode.bokeh:
            intruder_select = bokeh.models.Select(
                title="",
                value=self._gold.patterns[0].phrase,
                options=[p.phrase for p in self._gold.patterns])
            intruder_free = bokeh.models.TextInput(value="", title="")

            query_tab1 = bokeh.models.Panel(child=bokeh.models.Div(text=""), title="no locator")
            query_tab2 = bokeh.models.Panel(child=intruder_select, title="fixed locator")
            query_tab3 = bokeh.models.Panel(child=intruder_free, title="free locator")
            query_tabs = bokeh.models.Tabs(tabs=[query_tab1, query_tab2, query_tab3])

            options_cb = bokeh.models.CheckboxButtonGroup(
                labels=["legend"], active=[0])
        else:
            intruder_select = widgets.Dropdown(
                value=[p.phrase for p in self._gold.patterns][0],
                options=[p.phrase for p in self._gold.patterns])
            intruder_free = widgets.Text(value="")
            
            query_tab1 = widgets.HTML("")
            query_tab2 = intruder_select
            query_tab3 = intruder_free
            query_tabs = widgets.Tab(children=[query_tab1, query_tab2, query_tab3])
            query_tabs.set_title(0, "no locator")
            query_tabs.set_title(1, "fixed locator")
            query_tabs.set_title(2, "free locator")
            
        source = dict((k, bokeh.models.ColumnDataSource(v)) for k, v in self._compute_source_data("").items())
        
        cmap = bokeh.transform.factor_cmap(
            'query',
            palette=bokeh.palettes.Category20[len(self._gold.patterns)],
            factors=[p.phrase for p in self._gold.patterns])

        if has_tok_emb:
            tok_emb_p = bokeh.plotting.figure(
                plot_width=int(plot_width * 0.4), plot_height=600,
                title=f"Token Embeddings",
                toolbar_location="right",
                tools="pan,wheel_zoom,box_zoom,reset",
                tooltips=self._tok_emb_tooltips,
                visible=True)
            plot_width = int(plot_width * 0.6)

            tok_emb_p.circle(
                source=source['tokens'],
                size=10,
                #legend_field='query',
                color=cmap,
                alpha=0.8)

            if _display_mode.bokeh:
                tok_emb_status = bokeh.models.Div(text="")
            else:
                tok_emb_status = widgets.Label(text="")
                #tok_emb_update = widgets.Button(description="update")

            token_labels = bokeh.models.LabelSet(x='x', y='y', text='token',
                x_offset=5, y_offset=5, source=source['tokens'],
                render_mode='canvas', text_font_size='6pt')
            tok_emb_p.add_layout(token_labels)            
        else:
            tok_emb_p = None
            tok_emb_status = None
            
        is_interactive = has_tok_emb and _display_mode.fully_interactive
        
        doc_emb_p = bokeh.plotting.figure(
            plot_width=plot_width, plot_height=600,
            title=f"Document Embeddings",
            toolbar_location="left" if has_tok_emb else "below",
            tools="pan, wheel_zoom, lasso_select, box_select" if is_interactive else "pan, wheel_zoom",
            active_drag="lasso_select" if is_interactive else "pan",
            tooltips=self._doc_emb_tooltips)
                
        doc_emb_p.circle(
            source=source['docs'],
            size=10,
            legend_field='query',
            color=cmap,
            alpha=0.8)
        
        doc_emb_p.circle_cross(
            source=source['query'],
            size=25,
            color=cmap,
            fill_alpha=0.5)
        
        legend = bokeh.models.Legend(items=doc_emb_p.legend.items, location="center")
        #doc_emb_p.add_layout(legend, 'right')
        doc_emb_p.legend.items = []        
        
        
        def set_tok_emb_status(status):
            if _display_mode.bokeh:
                if status == "ok":
                    if tok_emb_p:
                        tok_emb_p.visible = True
                    if tok_emb_status:
                        tok_emb_status.visible = False
                else:
                    if tok_emb_p:
                        tok_emb_p.visible = False
                    if tok_emb_status:
                        tok_emb_status.visible = True
                        tok_emb_status.text = f"""<p style="width:100%; font-weight: bold; text-align:center;">{status}</p>"""
            else:
                fig = self._figures_html[1] if self._figures_html else None
                if status == "ok":
                    if fig:
                        fig.layout.display = 'block'
                    if tok_emb_status:
                        tok_emb_status.layout.display = 'none'
                else:
                    if fig:
                        fig.layout.display = 'none'
                    if tok_emb_status:
                        tok_emb_status.layout.display = 'block'
                        tok_emb_status.text = f"""<p style="width:100%; font-weight: bold; text-align:center;">{status}</p>"""                
        set_tok_emb_status("")
                        
        def update_token_plot(max_token_count=750):
            selected = source['docs'].selected.indices
            self._current_selection = [self._docs[i].doc.unique_id for i in selected]
 
            if tok_emb_p is None:
                return
            
            embedding = self._embedder.encoder.embedding
            if embedding is None:
                clear_token_plot()
                set_tok_emb_status("No token embedding.")
                return

            if not selected:
                clear_token_plot()
                set_tok_emb_status("No selection.")
                return
                        
            token_embedding_data = []
            
            for i in selected:
                doc_data = self._docs[i]
                for span in doc_data.doc.spans(self._embedder.partition):
                    texts = [token.text for token in span]
                    for i, token in enumerate(span):
                        token_embedding_data.append({
                            'token': token,
                            'query': doc_data.query,
                            'text': texts[i],
                            'work': doc_data.work,
                            'context': ' '.join(texts[:i] + [
                                '<span style="font-weight:bold;">',
                                texts[i],
                                '</span>'] + texts[i + 1:])
                        })
                        
            if len(token_embedding_data) > max_token_count:
                clear_token_plot()
                set_tok_emb_status("Selection is too large.<br>Please select fewer documents.")
                return
                                    
            token_embedding_vecs = np.array(self.session.word_vec(
                embedding, [x['token'] for x in token_embedding_data]))

            mag = np.linalg.norm(token_embedding_vecs, axis=1)
            mask = mag >= 1e-4

            if np.sum(mask) < 1:
                clear_token_plot()
                set_tok_emb_status("No non-zero token embeddings found.")
                return

            token_embedding_vecs = token_embedding_vecs[mask]
            token_embedding_data = [token_embedding_data[i] for i in np.nonzero(mask)[0]]
            
            if np.any(np.isnan(token_embedding_vecs)):
                clear_token_plot()
                set_tok_emb_status("Token embeddings contained NAN values.")
                return

            try:
                X = self._tok_tsne.fit(token_embedding_vecs)
            except:
                clear_token_plot()
                set_tok_emb_status("An error occured inside TSNE.")
                return
            
            source['tokens'].data = {
                'x': X[:, 0],
                'y': X[:, 1],
                'query': [x['query'] for x in token_embedding_data],
                'token': [x['text'] for x in token_embedding_data],
                'work': [x['work'] for x in token_embedding_data],
                'context': [x['context'] for x in token_embedding_data]
            }
            
            set_tok_emb_status("ok")
            
            if not _display_mode.fully_interactive:
                self._update_figures_html()
            
        def toggle_legend(attr, old, new):
            if 0 in options_cb.active:
                legend.visible = True
            else:
                legend.visible = False
                
        def update_document_embedding_plot():
            if _display_mode.bokeh:
                active = query_tabs.active
            else:
                active = query_tabs.selected_index                
            
            if active == 0:
                intruder = ""
            else:
                intruder = [intruder_select, intruder_free][active - 1].value
            for k, v in self._compute_source_data(intruder).items():
                source[k].data = v
            if _display_mode.static:
                update_token_plot()

            if not _display_mode.fully_interactive:
                self._update_figures_html()
                
        def encoder_changed():
            update_document_embedding_plot()
            if selection:
                id_to_index = dict((doc_data.doc.unique_id, i) for i, doc_data in enumerate(self._docs))
                source['docs'].selected.indices = [id_to_index[x] for x in selection]
            
        def clear_token_plot():
            if tok_emb_p is None:
                return

            source['tokens'].data = self._empty_token_data
            set_tok_emb_status("")
            self._current_selection = None
                        
        def trigger_token_plot_update(tok_plot_state):
            if self._tok_plot_state == tok_plot_state:
                update_token_plot()
                            
        if _display_mode.bokeh:
            options_cb.visible = False  # broken in bokeh
            
            def update_document_embedding_plot_shim(attr, old, new):
                update_document_embedding_plot()

            if _display_mode.static:
                # self._embedder.disabled = True
                #embedding_select.disabled = True
                intruder_select.disabled = True
                intruder_free.disabled = True
                query_tabs.disabled = True
                #for x in query_tabs.tabs:
                #    x.disabled = True
                options_cb.disabled = True
            else:
                self._embedder.on_change(encoder_changed)
                #embedding_select.on_change("value", update_document_embedding_plot_shim)
                intruder_select.on_change("value", update_document_embedding_plot_shim)
                intruder_free.on_change("value", update_document_embedding_plot_shim)
                query_tabs.on_change("active", update_document_embedding_plot_shim)

                options_cb.on_change("active", toggle_legend)
        else:
            def update_document_embedding_plot_shim(changed):
                update_document_embedding_plot()

            self._embedder.on_change(encoder_changed)
            #embedding_select.observe(update_document_embedding_plot_shim, names="value")
            intruder_select.observe(update_document_embedding_plot_shim, names="value")
            intruder_free.observe(update_document_embedding_plot_shim, names="value")
            query_tabs.observe(update_document_embedding_plot_shim, names="selected_index")

        if _display_mode.bokeh:
            def selection_change(attr, old, new):
                clear_token_plot()
                set_tok_emb_status("Computing. Please Wait...")
                self._tok_plot_state += 1
                bokeh_doc.add_timeout_callback(functools.partial(
                    trigger_token_plot_update, self._tok_plot_state), 500)

            if not _display_mode.static:
                source['docs'].selected.on_change('indices', selection_change)
        else:
            '''
            source['docs'].selected.js_on_change('indices', bokeh.models.CustomJS(code="""
                var indices = cb_obj.indices;
                window.global_hack = indices; // FIXME
                //console.log("put", window.global_hack);
            """))
            
            self._selected_indices = []
            
            def set_selection(data):
                # print("?", set_selection, data)
                self._selected_indices = data
            
            def on_update(changed):
                self._pw.js_init("""
                    //console.log("get", window.global_hack);
                    set_selection(window.global_hack); // FIXME
                """, set_selection=set_selection)
                
                #print(self._selected_indices)
                source['docs'].selected.indices = self._selected_indices
                update_token_plot()
            
            # tok_emb_update.on_click(on_update)
            '''
        
        source['docs'].js_on_change("data", bokeh.models.CustomJS(args={'p': doc_emb_p}, code="""
            p.reset.emit();
        """))
        if tok_emb_p is not None:
            source['tokens'].js_on_change("data", bokeh.models.CustomJS(args={'p': tok_emb_p}, code="""
                p.reset.emit();
            """))
            
        if selection:
            id_to_index = dict((doc_data.doc.unique_id, i) for i, doc_data in enumerate(self._docs))
            source['docs'].selected.indices = [id_to_index[x] for x in selection]
            
            if not _display_mode.fully_interactive:
                update_token_plot()

            elif _display_mode.static:
                update_token_plot()
                
        if locator is not None:
            def set_active_query_tab(active):
                if _display_mode.bokeh:
                    query_tabs.active = active
                else:
                    query_tabs.selected_index = active

            if isinstance(locator, str):
                locator = ("free", locator)
            locator_type, locator_s = locator
            if locator_type == "fixed":
                phrases = [p.phrase for p in self._gold.patterns]
                intruder_select.value = phrases[find_index_by_filter(
                    phrases, locator_s)]
                set_active_query_tab(1)
            elif locator_type == "free":
                intruder_free.value = locator_s
                set_active_query_tab(2)
            else:
                raise ValueError(locator_type)
                
        if _display_mode.static:
            update_document_embedding_plot()
        
        if _display_mode.bokeh:
            if tok_emb_p is not None:
                figure_widget = bokeh.layouts.row(
                    doc_emb_p,
                    bokeh.layouts.column(tok_emb_status, tok_emb_p))
            else:
                figure_widget = doc_emb_p

            return bokeh.layouts.column(
                self._embedder.widget,
                bokeh.layouts.column(query_tabs, background="#F0F0F0"),
                figure_widget,
                options_cb,
                sizing_mode="stretch_width")
        else:
            self._figures = []
            self._figures_html = []

            self._figures.append(doc_emb_p)
            self._figures_html.append(widgets.HTML(""))

            if tok_emb_p is not None:
                self._figures.append(tok_emb_p)
                self._figures_html.append(widgets.HTML(""))
            
            if len(self._figures_html) == 2:
                figure_widget = widgets.HBox([
                    self._figures_html[0],
                    widgets.VBox([tok_emb_status, self._figures_html[1]])])
            else:
                figure_widget = self._figures_html[0]

            self._pw = jp_proxy_widget.JSProxyWidget()
            self._pw.element.empty()
            display(self._pw)
            
            root_widgets = [
                self._embedder.widget,
                query_tabs,
                figure_widget
            ]
            
            if tok_emb_p:
                root_widgets.append(make_limited_function_warning_widget(
                    "change the document selection on the left side"))
                                        
            return widgets.VBox(root_widgets)
                

                
def plot_doc_embeddings(embedder_factory, gold, plot_args):
    plotters = []
    
    for args in plot_args:
        plotter = EmbeddingPlotter(
            embedder_factory.create(args.get("encoder")),
            gold)
        plotters.append(plotter)
        
    def clean_kwargs(kwargs):
        kwargs = kwargs.copy()
        if "encoder" in kwargs:
            del kwargs["encoder"]
        return kwargs
    
    plot_width = default_plot_width() // len(plot_args)

    if _display_mode.bokeh:
        def mk_root(bokeh_doc):
            widgets = []
            for plotter, kwargs in zip(plotters, plot_args):
                widgets.append(plotter.mk_plot(
                    bokeh_doc, plot_width=plot_width, **clean_kwargs(kwargs)))
            return bokeh.layouts.row(widgets)

        if _display_mode.static:
            bokeh.io.show(mk_root(None))
        else:
            def add_root(bokeh_doc):
                bokeh_doc.add_root(mk_root(bokeh_doc))

            bokeh.io.show(add_root)
    else:
        plots = [
            plotter.mk_plot(None, plot_width=plot_width, **clean_kwargs(kwargs))
            for plotter, kwargs in zip(plotters, plot_args)]
        display(widgets.HBox(plots))
        for p in plotters:
            p._update_figures_html()
        
    return plotters
            
            
class DocEmbeddingExplorer:
    def __init__(self, *args, gold, **kwargs):
        self._embedder_factory = DocEmbedderFactory(*args, **kwargs)
        self._gold = gold
        
    def plot(self, args):
        return plot_doc_embeddings(
            self._embedder_factory,
            self._gold,
            args)

        
class TokenSimilarityPlotter:
    def _create_data(self, doc, ref_token, embedding):        
        token_sim = TokenSimilarity(
            self._session.embeddings[embedding].factory,
            CosineSimilarity())
        sim = partial(self._session.similarity, token_sim)
        is_ctx = any(e.is_contextual for e in token_sim.embeddings)
        
        data = collections.defaultdict(list)
        seen = set()

        for span in doc.spans(self._partition):
            for k, token in enumerate(span):
                s = None

                if is_ctx:  # contextual embeddings
                    s = sim(token, ref_token)
                    text = f"{token.text} [{k}]"
                else:  # static embeddings
                    if token.text not in seen:
                        s = sim(token.text, ref_token.text)
                        seen.add(token.text)
                        text = token.text

                if s is not None:
                    data['token'].append(text)
                    data['sim'].append(max(0, s))
                    
        data['sim'] = np.array(data['sim'])
        order = np.argsort(data['sim'])[::-1]
        
        order = order[:self._top_n]
        
        data['token'] = [data['token'][i] for i in order]
        data['sim'] = data['sim'][order]
        
        return dict((k, v) for k, v in data.items())

    def _create_state(self):
        query = DummyIndex(self._partition).make_query(self._token_text.value)
        query = prepare_docs([query], self._nlp)[0]
        ref_token = list(query.spans(self._partition))[0][0]
        
        doc = self._doc_id_to_doc[self._doc_select.value]

        data = []
        for figure in self._figures:
            data.append(self._create_data(
                doc, ref_token, figure['embedding_select'].value))

        return {
            'label': f'top {self._top_n} similarities to "{ref_token.text}"',
            'data': data
        }
    
    def _update(self, index=None):
        state = self._create_state()
        
        for i, (figure, data) in enumerate(zip(self._figures, state["data"])):
            if index is not None and i != index:
                continue
            
            figure['source'].data = data
            p = figure['figure']
            #p.title.text = state["title"]

            p.y_range.factors = list(reversed(data["token"]))
            p.plot_height = len(data["token"]) * self._height_per_token

        p = self._figures[0]['figure']
        p.yaxis.axis_label = state['label']
        
        if not _display_mode.fully_interactive:
            self._update_figures_html()
            
    def __init__(self, session, nlp, gold, token, initial_occ=None, n_figures=2, top_n=15):
        self._session = session
        self._nlp = nlp
        self._gold = gold

        #self._occs = dict((x.unique_id, x) for x in gold.occurrences)
        #doc_digests = sorted(gold.doc_digests(), key=lambda x: x[0])
        self._doc_id_to_doc = dict((x.unique_id, x) for x in self._session.documents)
        self._embedding_names = sorted(session.embeddings.keys(), key=lambda x: len(x))
        
        if initial_occ is None:
            initial_occ_id = gold.occurrences[0].unique_id
        else:
            initial_occ_id = initial_occ.unique_id
        
        self._figures = None
        self._n_figures = min(n_figures, len(session.embeddings))
        self._height_per_token = 20
        self._top_n = top_n
        
        if _display_mode.bokeh:
            self._token_text = bokeh.models.TextInput(value=token)
            self._doc_select = bokeh.models.Select(
                options=[(occ.unique_id, occ_digest(occ)) for occ in gold.occurrences], value=initial_occ_id)
            #self._top_n = bokeh.models.Slider(start=5, end=100, step=5, value=15, title="top n")

            if _display_mode.static:
                self._token_text.disabled = True
                self._doc_select.disabled = True
            else:
                self._token_text.on_change("value", lambda attr, old, new: self._update())
                self._doc_select.on_change("value", lambda attr, old, new: self._update())
            #self._top_n.on_change("value", lambda attr, old, new: self._update())
        else:
            self._token_text = widgets.Text(value=token)
            self._doc_select = widgets.Dropdown(
                options=[(occ_digest(occ), occ.unique_id) for occ in gold.occurrences],
                value=initial_occ_id)
            self._token_text.observe(lambda change: self._update(), names='value')
            self._doc_select.observe(lambda change: self._update(), names='value')

        self._partition = session.partition("document")
        
        self._color_mapper = bokeh.models.LinearColorMapper(
            palette="Viridis256", low=0, high=1)
        
    def _create_figure_record(self, index):
        if _display_mode.bokeh:
            embedding_select = bokeh.models.Select(
                options=self._embedding_names, value=self._embedding_names[index])
            if _display_mode.static:
                embedding_select.disabled = True
            else:
                embedding_select.on_change("value", lambda attr, old, new: self._update(index=index))
        else:
            embedding_select = widgets.Dropdown(
                options=self._embedding_names, value=self._embedding_names[index])
            embedding_select.observe(lambda change: self._update(index=index), names='value')
        
        return {
            'source': None,
            'figure': None,
            'embedding_select': embedding_select
        }
    
    def _init_figure_record(self, figure, data):
        source = bokeh.models.ColumnDataSource(data)
        
        tooltips = """
            @sim
        """
        
        total_width = default_plot_width()

        p = bokeh.plotting.figure(
            y_range=list(reversed(data["token"])),
            plot_width=int(total_width / self._n_figures),
            plot_height=len(data["token"]) * self._height_per_token,
            title="",
            toolbar_location=None, tools="", tooltips=tooltips)
        
        p.hbar(
            "token", right="sim",
            source=source, height=0.5,
            color={'field': 'sim', 'transform': self._color_mapper})

        p.x_range = bokeh.models.Range1d(0, 1)
        p.ygrid.grid_line_color = None
        
        figure['source'] = source
        figure['figure'] = p
        
    def _configure_figures(self, state):
        p = self._figures[0]['figure']
        p.yaxis.axis_label = state['label']
        for x in self._figures:
            p = x['figure']
            p.xaxis.axis_label = "cosine similarity"
            
    def _update_figures_html(self):
        script, divs = bokeh.embed.components([x['figure'] for x in self._figures])
        for html, div in zip(self._figures_html, divs):
            html.value = div

        for x in BeautifulSoup(script, features="html.parser").findAll("script"):
            self._pw.js_init(x.string)
        
    def create(self, bokeh_doc):
        self._figures = [self._create_figure_record(i) for i in range(self._n_figures)]
        
        state = self._create_state()

        for figure, data in zip(self._figures, state['data']):
            self._init_figure_record(figure, data)

        self._configure_figures(state)
            
        if bokeh_doc or _display_mode.static:
            root = bokeh.layouts.column(
                self._token_text,
                self._doc_select,
                bokeh.layouts.row(*[
                    bokeh.layouts.column(x['embedding_select'], x['figure']) for x in self._figures]))
                        
            if _display_mode.static:
                bokeh.io.show(root)
            else:
                bokeh_doc.add_root(root)
            
        else:
            self._figures_html = [widgets.HTML("") for _ in self._figures]
            
            columns = [widgets.VBox([x['embedding_select'], html]) for x, html in zip(self._figures, self._figures_html)]
            display(widgets.VBox([self._token_text, self._doc_select, widgets.HBox(columns)]))
            
            self._pw = jp_proxy_widget.JSProxyWidget()
            self._pw.element.empty()
            display(self._pw)
            
            self._update_figures_html()

            
def plot_token_similarity(session, nlp, gold, token="high", occ=None, n_figures=2, top_n=15):
    plotter = TokenSimilarityPlotter(session, nlp, gold, token, occ, n_figures=n_figures, top_n=top_n)
    if _display_mode.fully_interactive:
        bokeh.io.show(plotter.create)
    else:
        plotter.create(None)
    
    
def dcg(rel, n):
    #return sum(rel[i - 1] / math.log2(i + 1) for i in range(1, n + 1))
    return np.sum(rel[:n] / np.log2(np.arange(2, n + 2)))

    
def ndcg(recommended, relevant, n):
    relevant = set(relevant)

    assert n <= len(recommended)
    
    if len(relevant) < 1:
        raise ValueError("need at least 1 relevant item")

    size = max(len(recommended), len(relevant))

    rel = np.zeros((size,), dtype=np.float32)
    for k, rec in enumerate(recommended):
        if rec in relevant:
            rel[k] = 1

    max_rel = np.zeros((size,), dtype=np.float32)
    max_rel[:len(relevant)] = 1

    return dcg(rel, n) / dcg(max_rel, n)


class NDCGComputer:
    def __init__(self, gold):
        self._gold = gold
        
        to_index = {}
        for occ in self._gold.occurrences:
            to_index[occ.unique_id] = len(to_index)
        self._to_index = to_index
        self._num_docs = len(to_index)

    def from_matches(self, matches, pattern):
        recommended = [m.doc.unique_id for m in matches]
        relevant = [x.unique_id for x in pattern.occurrences]
        return ndcg(recommended, relevant, len(recommended))
    
    def from_index(self, index, pattern):
        k = self._num_docs  # i.e. return ranking of full corpus
        result = index.find(pattern.phrase, n=k, disable_progress=True)
        return self.from_matches(result.matches, pattern)

    
class NDCGPlotter:
    def __init__(self, gold):
        self._ndcg = NDCGComputer(gold)

        self._gold = gold
        phrase = ([f"mean NDCG"] + [p.phrase for p in self._gold.patterns])[::-1]
        self._phrase = phrase

        p = bokeh.plotting.figure(
            y_range=phrase, plot_width=1000, plot_height=20 * len(self._gold.patterns),
            title="",
            toolbar_location=None, tools="")
        p.x_range = bokeh.models.Range1d(0, 1)
        p.ygrid.visible = False
        p.xaxis.axis_label = 'NDCG'

        self._p = p        
        self._bokeh_handle = None

    def _ndcg_array(self, index, pbar):
        ndcg = []
        for p in self._gold.patterns:
            ndcg.append(self._ndcg.from_index(index, p))
            pbar.update(1)
        return ([np.average(ndcg)] + ndcg)[::-1]
    
    def _format_ndcg(self, ndcg):
        return ['%.1f%%' % (x * 100) for x in ndcg]
            
    def update_ungrouped(self, index):
        with tqdm(total=len(self._gold.patterns)) as pbar:
            ndcg = self._ndcg_array(index, pbar)
        
        if self._bokeh_handle is None:
            self._source = bokeh.models.ColumnDataSource({
                'phrase': self._phrase,
                'ndcg': ndcg,
                'ndcg_str': self._format_ndcg(ndcg),
                'color': [1] * (len(self._phrase) - 1) + [0]
            })

            mapper = bokeh.transform.linear_cmap(
                field_name='color', palette=bokeh.palettes.Set2[3], low=0, high=1)

            self._p.hbar(
                'phrase', right='ndcg', color=mapper,
                source=self._source, height=0.75)
            
            labels = bokeh.models.LabelSet(x='ndcg', y='phrase', text='ndcg_str', level='glyph',
                x_offset=0, y_offset=0, source=self._source, render_mode='canvas',
                text_font_size='8pt', text_align='right', text_baseline='middle', text_color='white')
            self._p.add_layout(labels)

            self._bokeh_handle = bokeh.io.show(self._p, notebook_handle=True)
        else:
            self._source.data['ndcg'] = ndcg
            self._source.data['ndcg_str'] = self._format_ndcg(ndcg)
            bokeh.io.push_notebook(handle=self._bokeh_handle)

    def update_grouped(self, named_indices):
        named_indices = list(named_indices.items())
        index_names = [x[0] for x in named_indices]
        indices = [x[1] for x in named_indices]
        
        palette = bokeh.palettes.Set2
        
        y = [(x, index_name) for x in self._phrase for index_name in index_names[::-1]]
        
        self._p = bokeh.plotting.figure(
            y_range=bokeh.models.FactorRange(*y),
            plot_width=1000, plot_height=20 * len(self._gold.patterns) * len(indices),
            title="",
            toolbar_location=None, tools="")
        self._p.x_range = bokeh.models.Range1d(0, 1)
        self._p.ygrid.visible = False
        self._p.xaxis.axis_label = 'NDCG'
        
        self._p.yaxis.group_label_orientation = 0
       
        with tqdm(total=len(self._gold.patterns) * len(indices)) as pbar:
            ndcg = np.array([self._ndcg_array(index, pbar) for index in indices])[::-1]
        flat_ndcg = np.transpose(ndcg).flatten()
        
        color = np.repeat(
            np.linspace(0, 1, len(indices))[np.newaxis],
            len(self._phrase), axis=0).flatten()

        self._source = bokeh.models.ColumnDataSource({
            'phrase': y,
            'ndcg': flat_ndcg,
            'ndcg_str': self._format_ndcg(flat_ndcg),
            'color': color,
            'y': np.array(list(range(len(self._phrase) * len(indices)))) * 2
        })

        mapper = bokeh.transform.linear_cmap(
            field_name='color', palette=palette[max(3, len(indices))], low=0, high=1)

        self._p.hbar(
            'phrase', right='ndcg', y='y', color=mapper,
            source=self._source, height=1)

        labels = bokeh.models.LabelSet(x='ndcg', y='phrase', text='ndcg_str', level='glyph',
            x_offset=0, y_offset=0, source=self._source, render_mode='canvas',
            text_font_size='8pt', text_align='right', text_baseline='middle', text_color='white')
        self._p.add_layout(labels)

        _bokeh_show(self._p)
            

def plot_ndcgs(gold, named_indices):
    plotter = NDCGPlotter(gold)
    plotter.update_grouped(named_indices)
    
    
class InteractiveQuery:
    def __init__(self, session, nlp, partition_encoders={}, strategy="Alignment", strategy_options={}):
        self._session = session
        self._nlp = nlp
        self._partition_encoders = partition_encoders
        self._ui = PartitionMetricWidget(self, default=strategy, default_options=strategy_options)
        self._summary_widget = widgets.HTML("")
    
    @property
    def session(self):
        return self._session
    
    @property
    def partition(self):
        return self._session.partition("document")
    
    @property
    def partition_encoders(self):
        return self._partition_encoders
    
    @property
    def widget(self):
        return self._ui.widget
    
    @property
    def summary_widget(self):
        return self._summary_widget
    
    def describe(self):
        return self._ui.describe()
    
    def on_changed(self):
        html = markdown.markdown(self.describe())
        html = html.replace("<strong>", "<b>")
        html = html.replace("</strong>", "</b>")
        html = f'<span style="line-height:normal;">{html}</span>'
        self._summary_widget.value = html
    
    def create_index(self):
        return self._session.partition("document").index(self._ui.make(), self._nlp)
    
    @property
    def ordered_embedding(self):
        return sorted(list(self._session.embeddings.items()), key=lambda x: x[0])

        
class InteractiveIndexBuilder:
    def __init__(self, session, nlp, partition_encoders={}, strategy="Alignment", strategy_options={}):
        query_ui = InteractiveQuery(
            session, nlp,
            partition_encoders=partition_encoders,
            strategy=strategy,
            strategy_options=strategy_options)
        query_ui.on_changed()
        self._query_ui = query_ui
        
        if False and _display_mode.static:
            self._displayable = IPython.core.display.HTML(f"""
                <div style="background-color:#E0F0E0; margin-left: 2em; padding: 0.4em;">
                    {query_ui.summary_widget.value}
                </div>
                """)
        else:
            tab = widgets.Tab(
                children=[query_ui.summary_widget, query_ui.widget])
            for i, title in enumerate(['Index Summary', 'Edit']):
                tab.set_title(i, title)
            self._displayable = tab
            
    
    @property
    def displayable(self):
        return self._displayable
    
    def _ipython_display_(self):
        display(self._displayable)
                    
    def build_index(self):
        return self._query_ui.create_index()
    
    
class ResultScoresPlotter:
    def __init__(self, gold, index, query=None):
        self._gold = gold
        self._index = index
        self._selected_rank = None

        self._doc_formatter = DocFormatter(gold)
        self._ndcg = NDCGComputer(gold)
        
        self._p = None
        self._source = None

        self._result = None
        self._result_html = None
        
        default_query = gold.patterns[0].phrase
        if query is not None:
            candidates = [x for x in gold.patterns if x.phrase.startswith(query)]
            if len(candidates) > 0:
                default_query = candidates[0].phrase
        self._default_query = default_query

        phrases = [p.phrase for p in gold.patterns]
        self._query_select = bokeh.models.Select(
            title='', options=phrases, value=self._default_query)
        
    @property
    def matches(self):
        return self._result.matches
        
    @property
    def selected_match(self):
        if self._selected_rank is None:
            return None
        else:
            return self._result.matches[self._selected_rank - 1]

    def _run_query(self):
        phrases = [p.phrase for p in self._gold.patterns]
        pattern = self._gold.patterns[phrases.index(self._query_select.value)]
        n = len(self._gold.occurrences)
        gold_matches = [x.unique_id for x in pattern.occurrences]
        result = self._index.find(pattern.phrase, n=n, disable_progress=True)
        self._result = result
            
        ndcg = self._ndcg.from_matches(result.matches, pattern)
        title = ""  #f"Scores for Query '{query['phrase']}', NDCG={ndcg * 100:.1f}%"            
            
        base_hue = [0.8 if m.doc.unique_id in gold_matches else 0.2 for m in result.matches]
            
        return {
            'title': title,
            'data': {
                'base_hue': base_hue,
                'hue': base_hue[:],   
                'rank': [str(i) for i in range(1, n + 1)],
                'score': [m.score for m in result.matches],
                'tooltip': [self._doc_formatter.format_doc(m.prepared_doc) for m in result.matches]
            }
        }
            
    def _select_rank(self, rank):
        source = self._source
        
        hue = source.data["base_hue"][:]
        hue[rank - 1] = 0.1 if hue[rank - 1] < 0.5 else 0.9
        source.data["hue"] = hue
              
        from vectorian.render.excerpt import ExcerptRenderer
        from vectorian.render.render import Renderer
        from vectorian.render.location import LocationFormatter
        
        renderer = Renderer(
            [ExcerptRenderer()],
            LocationFormatter())
        
        html = renderer.to_html([self._result.matches[rank - 1]])
        self._result_html.value = html
        self._selected_rank = rank
            
    def _on_tap(self, event):
        i = math.floor(event.x)
        self._select_rank(i + 1)
        
    def on_update(self):
        qr = self._run_query()
        self._p.title.text = qr["title"]
        self._source.data = qr["data"]
        self._result_html.value = ""
        
    def build(self, rank=None, plot_width=None, plot_height=250):
        if plot_width is None:
            plot_width = default_plot_width()
        
        tooltips = """
            @tooltip
        """

        if _display_mode.fully_interactive:
            self._query_select.on_change('value', lambda attr, old, new: self.on_update())
        else:
            self._query_select.disabled = True
        
        qr = self._run_query()

        p = bokeh.plotting.figure(
            x_range=qr['data']['rank'], plot_width=plot_width, plot_height=plot_height,
            title=qr['title'],
            toolbar_location=None, tools="", tooltips=tooltips)
        self._p = p

        p.xaxis.axis_label = 'rank'
        p.yaxis.axis_label = 'NDCG'
        
        if _display_mode.fully_interactive:
            p.on_event(bokeh.events.Tap, self._on_tap)
        
        source = bokeh.models.ColumnDataSource(qr["data"])
        self._source = source

        mapper = bokeh.transform.linear_cmap(
            field_name='hue', palette=bokeh.palettes.RdBu6, low=0, high=1)
        vbar = p.vbar(
            "rank", top="score", color=mapper, source=source, width=0.8)

        #p.y_range = bokeh.models.Range1d(0, 1)
        p.xaxis.major_label_orientation = np.pi / 2
        p.xgrid.visible = False

        self._result_html = widgets.HTML("")

        if rank:
            self._select_rank(rank)
        
        bokeh_widget = bokeh.layouts.column(self._query_select, p)
                
        return bokeh_widget, self._result_html
                        
            
def plot_results(gold, index, query=None, rank=None, plot_height=200):
    bks = []
    jps = []
    plot_width = 1200
    
    drills = [dict(query=query, rank=rank)]
    plotters = []
    
    for drill in drills:
        rank = drill.get("rank")
        plotter = ResultScoresPlotter(gold, index, query)
        plotters.append(plotter)
        bk, jp = plotter.build(rank, plot_width=plot_width // len(drills), plot_height=plot_height)
        bks.append(bk)
        jps.append(jp)
        
    bk_root = bokeh.layouts.row(*bks)
    
    if _display_mode.fully_interactive:
        bokeh.io.show(lambda doc: doc.add_root(bk_root))
    else:
        bokeh.io.show(bk_root)

    result_widgets = [widgets.HBox(jps, layout=widgets.Layout(width=f'{plot_width}px'))] if len(jps) > 1 else jps[:1]
    if not _display_mode.fully_interactive:
        result_widgets.append(make_limited_function_warning_widget(
            "change the selected query and the selected rank"))
        
    if _display_mode.static:
        for jp in jps:
            display(IPython.core.display.HTML(jp.value))
    else:
        display(widgets.VBox(result_widgets))
    
    return plotters[0]

        
def plot_gold(gold):
    G = nx.Graph()

    color = {}
    subset = {}
    
    phrase = {}
    context = {}
    
    formatter = OccurenceFormatter()
    palette = bokeh.palettes.Spectral4

    doc_template = string.Template("""
        <div style="margin-left:2em">
            <span style="font-variant:small-caps; font-size: 14pt;">${title}</style>
            <hr>
            <div style="font-variant:normal; font-size: 10pt;">${text}</div>
        </div>
        """)
    
    for i, pattern in enumerate(gold.patterns):
        G.add_node(pattern.phrase)
        color[pattern.phrase] = palette[0]
        subset[pattern.phrase] = i
        
        phrase_html = f'<i>{pattern.phrase}</i>'
        phrase[pattern.phrase] = phrase_html
        context[pattern.phrase] = ""

        for occ in pattern.occurrences:
            phrase[occ.unique_id] = ""  # phrase_html + "<hr>"
            context[occ.unique_id] = doc_template.substitute(
                title=occ.source.work,
                phrase=occ.evidence.phrase,
                text=formatter.format_occurrence(occ))

            G.add_edge(pattern.phrase, occ.unique_id)
            color[occ.unique_id] = palette[1]
            subset[occ.unique_id] = i

    nx.set_node_attributes(G, color, "node_color")
    nx.set_node_attributes(G, subset, "subset")
    nx.set_node_attributes(G, phrase, "phrase")
    nx.set_node_attributes(G, context, "context")
            
    fixed = []
    pos = {}
    for i, pattern in enumerate(gold.patterns):
        fixed.append(pattern.phrase)
        y = i // 8
        x = i % 8
        s = 0.75
        pos[pattern.phrase] = (x * s, y * s)

    pos_arr = np.array(list(pos.values()))
    pad = 1
    
    plot_width = default_plot_width()
        
    plot = bokeh.models.Plot(
        plot_width=plot_width, plot_height=400,
        x_range=bokeh.models.Range1d(np.min(pos_arr[:, 0]) - pad, np.max(pos_arr[:, 0]) + pad),
        y_range=bokeh.models.Range1d(np.min(pos_arr[:, 1]) - pad, np.max(pos_arr[:, 1]) + pad),
        output_backend="svg")
    
    node_hover_tool = bokeh.models.HoverTool(
        tooltips="""
        @phrase
        @context
        """)
    plot.add_tools(node_hover_tool)
    
    graph_renderer = bokeh.plotting.from_networkx(
        G, nx.spring_layout, fixed=fixed, pos=pos, scale=0.5, k=0.15, center=(0, 0), iterations=100)
    graph_renderer.node_renderer.glyph = bokeh.models.Circle(size=10, fill_color="node_color")
    graph_renderer.edge_renderer.glyph = bokeh.models.MultiLine(line_color="black", line_alpha=1, line_width=1.5)
    plot.renderers.append(graph_renderer)

    '''
    token_labels = bokeh.models.LabelSet(x='x', y='y', text='token',
        x_offset=5, y_offset=5, source=source['tokens'],
        render_mode='canvas', text_font_size='6pt')
    tok_emb_p.add_layout(token_labels)
    '''
    
    bokeh.io.show(plot)
    

def get_token_scores_s(match):
    for region in match.regions():
        #print(region.gap_penalty, region)
        if region.match and region.match.edges:
            # use top weighted edge, if more than one.
            d = min([e.distance for e in region.match.edges])
            yield region.s, 1 - d
    

def get_token_scores_t(match):
    for region in match.regions():
        #print(region.gap_penalty, region)
        if region.match and region.match.edges:
            # use top weighted edge, if more than one.
            i = np.argmin([e.distance for e in region.match.edges])
            edge = region.match.edges[i] 
            d = edge.distance
            t = edge.t
            yield (t.text, t.index), 1 - d


def score_summary(match, get_scores=get_token_scores_s):
    # Vectorian computes its alignment scores by summing the query token's scores (each
    # between 0 and 1), then subtracting gap penalties, and then normalizing so that the
    # resulting score is between 0 and 1 (in the case of untagged alignments, this last
    # step simply means dividing by n, if n is the nnumber of matched query tokens).

    abs_scores = list(get_scores(match))

    gap_penalties = []
    for region in match.regions():
        if region.gap_penalty > 0:
            gap_penalties.append(region.gap_penalty)

    max_score = match.score_max

    # we produce token scores from which we subtracted the
    # global gap penalty. this allows us to return the gap
    # penalty as a positive term, instead of a negative
    # term - which helps with plotting this data.
    
    base_score = sum(x[1] for x in abs_scores)
    c = (base_score - sum(gap_penalties)) / base_score
    tokens = [(k, v * c) for k, v in abs_scores]
    
    data = {
        'tokens': tokens,
        'gaps': gap_penalties,
        'rest': max_score - (sum(x[1] for x in tokens) + sum(gap_penalties))
    }
    
    return {
        'tokens': [(k, v / max_score) for k, v in data['tokens']],
        'gaps': [x / max_score for x in gap_penalties],
        'rest': data['rest'] / max_score
    }


class TokenColors:
    def __init__(self):
        import bokeh.palettes

        self._colors = bokeh.palettes.Category20[20]

        token_color_indices = [i for i in range(20) if i not in (6, 15)]
        self._token_colors = [self._colors[i] for i in token_color_indices]
    
    def tokens(self, n):
        return list(itertools.islice(itertools.cycle(self._token_colors), None, n))
    
    @property
    def gap(self):
        return self._colors[6]

    @property
    def err(self):
        return self._colors[15]


def token_scores_stacked_bar_chart(matches, ranks=None, highlight=None, show_gap_penalty=True, plot_width=800):
    if ranks is None:
        ranks = list(range(1, len(matches) + 1))

    data = collections.defaultdict(list)
    tokens = set()

    summaries = []
    for j in ranks:
        match = matches[j - 1]
        summary = score_summary(match, get_token_scores_t)
        summaries.append(summary)
        for k, v in summary['tokens']:
            tokens.add(k)
            
        #x_sum = sum([x[1] for x in summary['tokens']])
        #print(j + 1, x_sum, match.score, math.isclose(x_sum, match.score, abs_tol=0.001), len(summary['tokens']))
        
    tokens = sorted(tokens, key=lambda x: x[1])
    
    def token_key(text, index):
        if index is not None:
            return f"{text} [{index}]"
        else:
            return text
    
    for j, summary in enumerate(summaries):
        match_tokens = dict(summary['tokens'])
        for t in tokens:
            x = match_tokens.get(t, 0)
            data[token_key(*t)].append(x)
        data["gap"].append(sum(summary['gaps']))
        data["x"].append(j + 1)
      
    p = bokeh.plotting.figure(
        plot_width=plot_width, plot_height=250,
        title="", toolbar_location=None)
    
    if show_gap_penalty:
        elements = [("gap", None)] + tokens
    else:
        elements = tokens

    line_color = ([None] * len(tokens)) + ([None] if show_gap_penalty else [])
    line_dash = ["solid"] * len(tokens) + (["solid"] if show_gap_penalty else [])
    if highlight and highlight.get("token"):
        token_text = [t[0] for t in elements[::-1]]
        for j, x in enumerate(highlight["token"]):
            i = token_text.index(x)
            line_color[i] = "black"
            line_dash[i] = ["solid", "dashed", "dotted"][j % 3]
        
    colors = TokenColors()
    vstack = p.vbar_stack(
        [token_key(*x) for x in elements[::-1]], x="x", width=0.9,
        color=colors.tokens(len(tokens)) + ([colors.gap] if show_gap_penalty else []),
        line_color=line_color, line_width=2, line_dash=line_dash,
        source=bokeh.models.ColumnDataSource(data))
    
    items = []
    for x, vs in zip(elements, vstack[::-1]):
        items.append((token_key(*x), [vs]))
    
    legend = bokeh.models.Legend(items=items, location=(0, -30))
    p.add_layout(legend, 'right')

    p.xgrid.visible = False
    p.xaxis.major_label_orientation = np.pi / 2
    p.xaxis.ticker = ranks
    
    if highlight and highlight.get("rank"):
        scores = [m.score for m in matches]
        x = []
        y = []
        for i in highlight["rank"]:
            x.append(i)
            y.append(scores[i - 1] + 0.05)    
        p.triangle(color="black", x=x, y=y, size=10, angle=np.pi)
    
    bokeh.io.show(p)
    
    
def token_scores_pie_chart(match, plot_size=350):
    summary = score_summary(match, get_token_scores_t)
    colors = TokenColors()

    data = {
        'element': [x[0][0] for x in summary['tokens']],
        'score': [x[1] for x in summary['tokens']],
        'color': colors.tokens(len(summary['tokens']))
    }
    
    data['element'].append('')
    data['score'].append(summary['rest'])
    data['color'].append(colors.err)
    
    for gap_penalty in summary['gaps']:
        data['element'].append('gap')
        data['score'].append(gap_penalty)
        data['color'].append(colors.gap)
    
    data['angle'] = np.array(data['score']) * (2 * np.pi * 0.8)
        
    data['end_angle'] = np.cumsum(data['angle'])
    data['start_angle'] = np.hstack([[0], data['end_angle']])[:-1]

    k = len(summary['gaps']) + 1
    data['start_angle'][-k:] += 2 * np.pi * 0.1
    data['end_angle'][-k:] += 2 * np.pi * 0.1

    source = bokeh.models.ColumnDataSource(data)
    
    p = bokeh.plotting.figure(
        plot_width=plot_size, plot_height=plot_size,
        title="", toolbar_location=None,
        tools="hover", tooltips="@element: @score")
    
    r1 = 0.25
    r2 = 0.5
    
    p.annular_wedge(
        x=0, y=0, inner_radius=r1, outer_radius=r2, direction="anticlock",
        start_angle='start_angle',
        end_angle='end_angle',
        line_color="white", fill_color='color', source=source)  # legend_field='element'
    
    label_angles = (data['start_angle'] + data['end_angle']) / 2
    #label_angles = [x * (np.pi / 180) for x in range(0, 360, 20)]
    
    def make_label_source(r):
        return bokeh.models.ColumnDataSource({
            'text': data['element'],
            'x': np.cos(label_angles) * r,
            'y': np.sin(label_angles) * r,
            'angle': label_angles
        })
    
    labels = bokeh.models.LabelSet(x='x', y='y', text='text',
        x_offset=0, y_offset=0, source=make_label_source((r1 + r2) / 2),
        text_font_size='10pt', angle='angle', render_mode='canvas',
        text_baseline='middle', text_align='center')
    p.add_layout(labels)

    #p.circle(x='x', y='y', source=make_label_source(0.3), color="black")
    
    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None

    plot_range = 0.55
    
    p.y_range.start = -plot_range
    p.y_range.end = plot_range

    p.x_range.start = -plot_range
    p.x_range.end = plot_range
    
    return p

    
def vis_token_scores(matches, kind="bar", ranks=None, highlight=None, plot_width=1000):
    if kind == "bar":
        if _display_mode.static:
            token_scores_stacked_bar_chart(
                matches,
                ranks=ranks,
                highlight=highlight,
                show_gap_penalty=False,
                plot_width=plot_width)
        else:
            @widgets.interact(indicate_gap_penalty=False)
            def plot(indicate_gap_penalty):
                token_scores_stacked_bar_chart(
                    matches,
                    ranks=ranks,
                    highlight=highlight,
                    show_gap_penalty=indicate_gap_penalty,
                    plot_width=plot_width)
            return plot
    elif kind == "pie":
        assert ranks is not None
        picked = [matches[i - 1] for i in ranks]
        n_cols = 3
        n_rows = int(np.ceil(len(picked) / n_cols))
        plot_size = plot_width // n_cols
        figures = [token_scores_pie_chart(m, plot_size=plot_size) for m in picked]
        bokeh.io.show(bokeh.layouts.gridplot(
            figures, ncols=n_cols, plot_width=plot_size, plot_height=plot_size * n_rows))
    else:
        raise ValueError(kind)

        
def plot_embedding_vectors(labels, vectors, palette, bg, extra_height=0, w_format="0.00"):    
    words = labels[::-1]
    vecs = np.array(vectors[::-1])

    dims = vecs.shape[-1]
    vecs = vecs.flatten()
    
    source = bokeh.models.ColumnDataSource({
        'x': np.array(list(range(dims)) * len(words)) + 1,
        'y': list(itertools.chain(*[[word] * dims for word in words])),
        'w': vecs
    })

    color_mapper = bokeh.models.LinearColorMapper(
        palette=palette, low=np.amin(vecs), high=np.amax(vecs))
    
    p = bokeh.plotting.figure(
        y_range=words,
        x_axis_type=None,
        x_range=(1 - 0.5, dims + 0.5),
        plot_width=default_plot_width(),
        plot_height=30 * len(words) + 20 + extra_height,
        title="",
        toolbar_location="below",
        tools="pan, wheel_zoom, box_zoom, reset",
        active_drag="box_zoom",
        tooltips=[("dim", "@x"), ("w", "@w{%s}" % w_format)])

    p.background_fill_color = "black"
    p.background_fill_alpha = bg
    
    p.square(source=source, size=10, color={'field': 'w', 'transform': color_mapper})

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    
    if dims <= 50:
        ticker = bokeh.models.SingleIntervalTicker(interval=5, num_minor_ticks=5)
    elif dims <= 500:
        ticker = bokeh.models.SingleIntervalTicker(interval=10, num_minor_ticks=2)
    else:
        ticker = bokeh.models.SingleIntervalTicker(interval=50, num_minor_ticks=2)
    xaxis = bokeh.models.LinearAxis(ticker=ticker)
    p.add_layout(xaxis, 'below')
    
    color_bar = bokeh.models.ColorBar(
        color_mapper=color_mapper, label_standoff=3, margin=20, height=5, padding=5)
    p.add_layout(color_bar, 'below')
    
    bokeh.io.show(p)
    
    
def _embedding_vectors(words, get_vec, normalize):
    if normalize:
        def get_norm_vec(word):
            v = get_vec(word)
            return v / np.linalg.norm(v)

        return np.array([get_norm_vec(word) for word in words])
    else:
        return np.array([get_vec(word) for word in words])        
    

def plot_embedding_vectors_val(words, get_vec, normalize=False):
    vecs = _embedding_vectors(words, get_vec, normalize)
    plot_embedding_vectors(words, vecs, "Viridis256", 0.7)
    

def plot_embedding_vectors_mul(pairs, get_vec):    
    words = []
    vecs = []
    for u, v in pairs:
        words.append(u + "-" + v)
        u_vec, v_vec = _embedding_vectors([u, v], get_vec, True)
        vecs.append([u_vec * v_vec])
    vecs = np.array(vecs)
    
    plot_embedding_vectors(words, vecs, "Inferno256", 1, 20, w_format="0.0000")

    
class DocEmbeddingBars:
    def __init__(self, embedder, session, gold_data):
        self._embedder = embedder
        self._gold_data = gold_data
        
        self._id_to_doc = dict((doc.unique_id, doc) for doc in session.documents)
        self._phrases = [x.phrase for x in gold_data.patterns]
    
    def plot_doc_emb(self, match_pattern, mismatch_pattern):
        match_i = self._phrases.index(match_pattern)
        mismatch_i = self._phrases.index(mismatch_pattern)

        items = {}
        doc_names = []

        items["pattern"] = self._embedder.mk_query(self._gold_data.patterns[match_i].phrase)
        for i, x in enumerate(self._gold_data.patterns[match_i].occurrences):
            name = f"match {i + 1}"
            items[name] = self._id_to_doc[x.unique_id]
            doc_names.append(name)
        for i, x in enumerate(self._gold_data.patterns[mismatch_i].occurrences):
            name = f"mismatch {i + 1}"
            items[name] = self._id_to_doc[x.unique_id]
            doc_names.append(name)

        pairs = []
        for name in doc_names:
            pairs.append(("pattern", name))

        plot_embedding_vectors_mul(
            pairs, get_vec=lambda w: self._embedder.encode([items[w]])[0])

    def plot(self, default_pattern, default_contrast):
        @widgets.interact(
            pattern=widgets.Dropdown(
                options=self._phrases,
                value=self._phrases[find_index_by_filter(self._phrases, default_pattern)],
                layout={'width': 'max-content'}),
            mismatch=widgets.Dropdown(
                options=self._phrases,
                value=self._phrases[find_index_by_filter(self._phrases, default_contrast)],
                layout={'width': 'max-content'}))
        def plot(pattern, mismatch):
            self.plot_doc_emb(pattern, mismatch)
