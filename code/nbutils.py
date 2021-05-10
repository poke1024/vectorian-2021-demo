import string
import collections
import numpy as np
import math
import os
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

import bokeh.plotting
import bokeh.models
import bokeh.transform
import bokeh.palettes
import bokeh.layouts

from functools import partial
from cached_property import cached_property
from IPython.core.display import HTML, display
from bs4 import BeautifulSoup

from vectorian.embeddings import TokenEmbeddingAggregator, prepare_docs
from vectorian.embeddings import CachedPartitionEncoder
from vectorian.index import DummyIndex
from vectorian.metrics import TokenSimilarity, CosineSimilarity
from vectorian.interact import PartitionMetricWidget


_has_bokeh_server = False

def running_inside_binder():
    return os.environ.get("BINDER_SERVICE_HOST") is not None


def initialize(has_bokeh_server="auto"):
    global _has_bokeh_server
    if has_bokeh_server == "auto":
        has_bokeh_server = not running_inside_binder()
    _has_bokeh_server = has_bokeh_server
    
    
initialize()

    
def make_limited_function_warning_widget(action_text):
    info_text = ("This visualization has limited interactivity due to technical constraints.<br>" +
        "Run this notebook on a local Jupyter installation (with Bokeh server support) in order to " +
        f"interactively {action_text}.")
    return widgets.HTML(
        value=f'<b><p style="text-align:center"><font color="red">{info_text}</p></b>',
        layout=widgets.Layout(border='solid 1px red'))

    
class Gold:
    def __init__(self, data):
        self._data = data
        
    @property
    def phrases(self):
        return [q["phrase"] for q in self._data]
    
    def matches(self, phrase):
        for q in self._data:
            if q["phrase"] == phrase:
                return q["matches"]
        return []

    @property
    def items(self):
        return self._data

    @cached_property
    def by_id(self):
        doc_details = {}

        for query in self._data:
            for m in query["matches"]:
                doc_details[m["id"]] = {
                    'query': query,
                    'match': m
                }

        return doc_details
    
    def doc_digests(self, n=80):
        for query in self._data:
            for m in query["matches"]:
                yield (f"{m['work']}: {m['context']}"[:n] + "..."), m['id']

    
class ContextFormatter:
    def __init__(self):
        pass
    
    def format_context(self, record):
        text = record["context"]
        quote = record["quote"]
        try:
            i = text.index(quote)
            return ''.join([
                text[:i],
                '<span style="font-weight:bold;">',
                text[i:i + len(quote)],
                '</span>',
                text[i + len(quote):]
            ])
        except:
            return text
            
    
class DocFormatter:
    def __init__(self, gold):
        self._gold = gold
        self._fmt = ContextFormatter()
        self._template = string.Template("""
            <div style="margin-left:2em">
                <span style="font-variant:small-caps; font-size: 14pt;">${title}</style>
                <span style="float:right; font-size: 10pt;">query: ${phrase}</span>
                <hr>
                <div style="font-variant:normal; font-size: 10pt;">${text}</div>
            </div>
            """)
       
    def format_context(self, doc):
        return self._fmt.format_context(
            self._gold.by_id[doc.unique_id]["match"])
        
    def format_doc(self, doc):
        return self._template.substitute(
            phrase=self._gold.by_id[doc.unique_id]["query"]["phrase"],
            title=doc.metadata["title"],
            text=self.format_context(doc))


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

    
def browse(gold, initial_phrase=1, initial_context=1, rows=5):
    formatter = ContextFormatter()
    records = gold.items
    
    if isinstance(initial_phrase, str):
        initial_phrase = find_index_by_filter(
            gold.phrases, initial_phrase) + 1
    
    query_select = widgets.Select(
        options=[(k, i) for i, k in enumerate(gold.phrases)],
        value=initial_phrase - 1,
        rows=rows,
        description='phrase:')
    
    def query_contexts():
        matches = records[query_select.value]["matches"]
        return [(x["work"], i) for i, x in enumerate(matches)]
    
    if isinstance(initial_context, str):
        initial_context = find_index_by_filter(
            [x[0] for x in query_contexts()], initial_context) + 1

    context_select = widgets.Select(
        options=query_contexts(),
        value=initial_context - 1,
        rows=rows,
        description='occurs in:')
    
    context_display = widgets.HTML("", description="as:")
    
    def on_phrase_change(change):
        context_select.unobserve(on_context_change)
        context_select.options = query_contexts()
        context_select.value = 0
        context_select.observe(on_context_change)
        on_context_change(None)

    def on_context_change(change):
        matches = records[query_select.value]["matches"]
        works = query_contexts()
        i = context_select.value
        context_display.value = formatter.format_context(matches[i])
        
    query_select.observe(on_phrase_change)
    context_select.observe(on_context_change)
    on_context_change(None)
    
    return widgets.HBox([query_select, context_select, context_display])    
    
    
class TSNECallback(openTSNE.callbacks.Callback):
    def optimization_about_to_start(self):
        pass

    def __call__(self, iteration, error, embedding):
        pass
    

class EmbeddingPlotter:    
    def __init__(self, session, nlp, gold, aggregator):
        self._session = session
        self._nlp = nlp
        self._gold = gold
        self._current_selection = None

        self._id_to_doc = dict((doc.unique_id, doc) for doc in self._session.documents)
        self._doc_formatter = DocFormatter(gold)
        
        DocData = collections.namedtuple("DocData", ["doc", "query", "work"])

        docs = []
        for q in self._gold.items:
            for m in q["matches"]:
                docs.append(DocData(
                    doc=self._id_to_doc[m["id"]],
                    query=q["phrase"],
                    work=m["work"]))
        self._docs = docs
        
        self._partition = session.partition("document")

        self.encoders = dict()
        for k, embedding in session.embeddings.items():
            self.encoders[format_embedding_name(k) + f" ({aggregator.__name__})"] = CachedPartitionEncoder(
                TokenEmbeddingAggregator(embedding.factory, aggregator))    
 
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
    def partition(self):
        return self._partition
    
    def _compute_source_data(self, embedding, intruder):        
        encoder = self.encoders[embedding]
        intruder_doc = DummyIndex(self.partition).make_query(intruder)
        
        id_to_doc = self._id_to_doc
        query_docs = []
        
        works = []
        phrases = []
        contexts = []

        query_docs.append(intruder_doc)
        works.append("")
        phrases.append(intruder)
        contexts.append("")

        for q in self._gold.items:
            for m in q["matches"]:
                doc = id_to_doc[m["id"]]
                query_docs.append(doc)
                works.append(m["work"])
                phrases.append(q['phrase'])
                contexts.append(self._doc_formatter.format_context(doc))

        data = {
            'work': works,
            'query': phrases,
            'context': contexts,
            'vector': encoder.encode(
                prepare_docs(query_docs, self._nlp), self.partition).unmodified
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
                
    def mk_plot(self, bokeh_doc, encoder=0, selection=[], locator=None, has_tok_emb=True):
        encoder_names = sorted(self.encoders.keys())
        
        if isinstance(encoder, str):
            encoder = find_index_by_filter(encoder_names, encoder)
        
        if _has_bokeh_server:
            embedding_select = bokeh.models.Select(
                title="",
                value=encoder_names[encoder],
                options=encoder_names,
                margin=(0, 20, 0, 0))

            intruder_select = bokeh.models.Select(
                title="",
                value=self._gold.phrases[0],
                options=self._gold.phrases)
            intruder_free = bokeh.models.TextInput(value="", title="")

            query_tab1 = bokeh.models.Panel(child=bokeh.models.Div(text=""), title="no locator")
            query_tab2 = bokeh.models.Panel(child=intruder_select, title="fixed locator")
            query_tab3 = bokeh.models.Panel(child=intruder_free, title="free locator")
            query_tabs = bokeh.models.Tabs(tabs=[query_tab1, query_tab2, query_tab3])

            options_cb = bokeh.models.CheckboxButtonGroup(
                labels=["legend"], active=[0])
        else:
            embedding_select = widgets.Dropdown(
                value=encoder_names[encoder],
                options=encoder_names)
            
            intruder_select = widgets.Dropdown(
                value=self._gold.phrases[0],
                options=self._gold.phrases)
            intruder_free = widgets.Text(value="")
            
            query_tab1 = widgets.HTML("")
            query_tab2 = intruder_select
            query_tab3 = intruder_free
            query_tabs = widgets.Tab(children=[query_tab1, query_tab2, query_tab3])
            query_tabs.set_title(0, "no locator")
            query_tabs.set_title(1, "fixed locator")
            query_tabs.set_title(2, "free locator")
            
        source = dict((k, bokeh.models.ColumnDataSource(v)) for k, v in self._compute_source_data(
            embedding_select.value, "").items())
        
        cmap = bokeh.transform.factor_cmap(
            'query',
            palette=bokeh.palettes.Category20[len(self._gold.phrases)],
            factors=self._gold.phrases)

        if has_tok_emb:
            tok_emb_p = bokeh.plotting.figure(
                plot_width=400, plot_height=600,
                title=f"Token Embeddings",
                toolbar_location="right",
                tools="pan,wheel_zoom,box_zoom,reset",
                tooltips=self._tok_emb_tooltips,
                visible=True)

            tok_emb_p.circle(
                source=source['tokens'],
                size=10,
                #legend_field='query',
                color=cmap,
                alpha=0.8)

            if _has_bokeh_server:
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
        
        doc_emb_p = bokeh.plotting.figure(
            plot_width=600, plot_height=600,
            title=f"Document Embeddings",
            toolbar_location="left" if has_tok_emb else "below",
            tools="pan, wheel_zoom, lasso_select, box_select" if (has_tok_emb and _has_bokeh_server) else "pan, wheel_zoom",
            active_drag="lasso_select" if (has_tok_emb and _has_bokeh_server) else "pan",
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
            if _has_bokeh_server:
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
            if tok_emb_p is None:
                return
            
            embedding = self.encoders[embedding_select.value].embedding
            if embedding is None:
                clear_token_plot()
                return

            selected = source['docs'].selected.indices
            if not selected:
                clear_token_plot()
                return
            
            self._current_selection = [self._docs[i].doc.unique_id for i in selected]
            
            token_embedding_data = []
            
            for i in selected:
                doc_data = self._docs[i]
                for span in doc_data.doc.spans(self._partition):
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
                                    
            token_embedding_vecs = np.array(self._session.word_vec(
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
            
            if not _has_bokeh_server:
                self._update_figures_html()
            
        def toggle_legend(attr, old, new):
            if 0 in options_cb.active:
                legend.visible = True
            else:
                legend.visible = False
                
        def update_document_embedding_plot():
            if _has_bokeh_server:
                active = query_tabs.active
            else:
                active = query_tabs.selected_index                
            
            if active == 0:
                intruder = ""
            else:
                intruder = [intruder_select, intruder_free][active - 1].value
            for k, v in self._compute_source_data(
                embedding_select.value, intruder).items():
                source[k].data = v
            update_token_plot()

            if not _has_bokeh_server:
                self._update_figures_html()
            
        def clear_token_plot():
            if tok_emb_p is None:
                return

            source['tokens'].data = self._empty_token_data
            set_tok_emb_status("")
            self._current_selection = None
                        
        def trigger_token_plot_update(tok_plot_state):
            if self._tok_plot_state == tok_plot_state:
                update_token_plot()
                            
        if _has_bokeh_server:
            options_cb.visible = False  # broken in bokeh
            
            def update_document_embedding_plot_shim(attr, old, new):
                update_document_embedding_plot()

            embedding_select.on_change("value", update_document_embedding_plot_shim)
            intruder_select.on_change("value", update_document_embedding_plot_shim)
            intruder_free.on_change("value", update_document_embedding_plot_shim)
            query_tabs.on_change("active", update_document_embedding_plot_shim)

            options_cb.on_change("active", toggle_legend)
        else:
            def update_document_embedding_plot_shim(changed):
                update_document_embedding_plot()

            embedding_select.observe(update_document_embedding_plot_shim, names="value")
            intruder_select.observe(update_document_embedding_plot_shim, names="value")
            intruder_free.observe(update_document_embedding_plot_shim, names="value")
            query_tabs.observe(update_document_embedding_plot_shim, names="selected_index")

        if _has_bokeh_server:
            def selection_change(attr, old, new):
                clear_token_plot()
                set_tok_emb_status("Computing. Please Wait...")
                self._tok_plot_state += 1
                bokeh_doc.add_timeout_callback(functools.partial(
                    trigger_token_plot_update, self._tok_plot_state), 500)

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
            
            if not _has_bokeh_server:
                update_token_plot()

        if locator is not None:
            if isinstance(locator, str):
                locator = ("free", locator)
            locator_type, locator_s = locator
            if locator_type == "fixed":
                intruder_select.value = self._gold.phrases[find_index_by_filter(
                    self._gold.phrases, locator_s)]
                query_tabs.active = 1
            elif locator_type == "free":
                intruder_free.value = locator_s
                query_tabs.active = 2
            else:
                raise ValueError(locator_type)
        
        if _has_bokeh_server:
            if tok_emb_p is not None:
                figure_widget = bokeh.layouts.row(
                    doc_emb_p,
                    bokeh.layouts.column(tok_emb_status, tok_emb_p))
            else:
                figure_widget = doc_emb_p

            return bokeh.layouts.column(
                bokeh.layouts.column(embedding_select, query_tabs, background="#F0F0F0"),
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
                widgets.VBox([embedding_select, query_tabs]),
                figure_widget
            ]
            
            if tok_emb_p:
                root_widgets.append(make_limited_function_warning_widget(
                    "change the document selection on the left side"))
                                        
            return widgets.VBox(root_widgets)
                

                
def plot_doc_embeddings(session, nlp, gold, plot_args, aggregator=np.mean, extra_encoders={}):
    plotters = []
    
    for args in plot_args:
        plotter = EmbeddingPlotter(session, nlp, gold, aggregator)
        for k, v in extra_encoders.items():
            plotter.encoders[k] = v
        plotters.append(plotter)

    if _has_bokeh_server:
        def add_root(bokeh_doc):
            widgets = []
            for plotter, kwargs in zip(plotters, plot_args):
                widgets.append(plotter.mk_plot(bokeh_doc, **kwargs))

            bokeh_doc.add_root(bokeh.layouts.row(widgets))

        bokeh.io.show(add_root)
    else:
        plots = [
            plotter.mk_plot(None, **kwargs)
            for plotter, kwargs in zip(plotters, plot_args)]
        display(widgets.HBox(plots))
        for p in plotters:
            p._update_figures_html()
        
    return plotters
            
            
class DocEmbeddingExplorer:
    def __init__(self, **base_args):
        self._aggregator = widgets.Dropdown(
            description="token embedding aggregator:",
            style = {'description_width': 'initial', 'width': 'max'},
            options=[("mean", np.mean), ("median", np.median), ("max", np.max), ("min", np.min)])
        display(self._aggregator)
        self._base_args = base_args
        
    def plot(self, args):
        return plot_doc_embeddings(
            plot_args=args,
            aggregator=self._aggregator.value,
            **self._base_args)

        
        
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
        
        if not _has_bokeh_server:
            self._update_figures_html()
            
    def __init__(self, session, nlp, gold, token, initial_doc=0, n_figures=2, top_n=15):
        self._session = session
        self._nlp = nlp
        self._gold = gold
        
        doc_digests = sorted(gold.doc_digests(), key=lambda x: x[0])
        self._doc_id_to_doc = dict((x.unique_id, x) for x in self._session.documents)
        self._embedding_names = sorted(session.embeddings.keys(), key=lambda x: len(x))
        
        if isinstance(initial_doc, str):
            i = find_index_by_filter([x[0] for x in doc_digests], initial_doc)
            initial_select_value = doc_digests[i][1]
        else:
            initial_select_value = doc_digests[initial_doc][1]

        self._figures = None
        self._n_figures = min(n_figures, len(session.embeddings))
        self._height_per_token = 20
        self._top_n = top_n
        
        if _has_bokeh_server:        
            self._token_text = bokeh.models.TextInput(value=token)
            self._doc_select = bokeh.models.Select(
                options=[(v, k) for k, v in doc_digests], value=initial_select_value)
            #self._top_n = bokeh.models.Slider(start=5, end=100, step=5, value=15, title="top n")

            self._token_text.on_change("value", lambda attr, old, new: self._update())
            self._doc_select.on_change("value", lambda attr, old, new: self._update())
            #self._top_n.on_change("value", lambda attr, old, new: self._update())
        else:
            self._token_text = widgets.Text(value=token)
            self._doc_select = widgets.Dropdown(options=doc_digests, value=initial_select_value)
            self._token_text.observe(lambda change: self._update(), names='value')
            self._doc_select.observe(lambda change: self._update(), names='value')

        self._partition = session.partition("document")
        
        self._color_mapper = bokeh.models.LinearColorMapper(
            palette="Viridis256", low=0, high=1)
        
    def _create_figure_record(self, index):
        if _has_bokeh_server:        
            embedding_select = bokeh.models.Select(
                options=self._embedding_names, value=self._embedding_names[index])
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

        p = bokeh.plotting.figure(
            y_range=list(reversed(data["token"])),
            plot_width=int(1000 / self._n_figures),
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
            
        if bokeh_doc:
            bokeh_doc.add_root(bokeh.layouts.column(
                self._token_text,
                self._doc_select,
                bokeh.layouts.row(*[
                    bokeh.layouts.column(x['embedding_select'], x['figure']) for x in self._figures])
            ))
        else:
            self._figures_html = [widgets.HTML("") for _ in self._figures]
            
            columns = [widgets.VBox([x['embedding_select'], html]) for x, html in zip(self._figures, self._figures_html)]
            display(widgets.VBox([self._token_text, self._doc_select, widgets.HBox(columns)]))
            
            self._pw = jp_proxy_widget.JSProxyWidget()
            self._pw.element.empty()
            display(self._pw)
            
            self._update_figures_html()

            
def plot_token_similarity(session, nlp, gold, token="high", doc=0, n_figures=2, top_n=15):
    plotter = TokenSimilarityPlotter(session, nlp, gold, token, doc, n_figures=n_figures, top_n=top_n)
    if _has_bokeh_server:
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
        for q in self._gold.items:
            for m in q["matches"]:
                to_index[m["id"]] = len(to_index)
        self._to_index = to_index
        self._num_docs = len(to_index)

    def from_matches(self, matches, query):
        recommended = [m.doc.unique_id for m in matches]
        relevant = [m["id"] for m in query["matches"]]
        return ndcg(recommended, relevant, len(recommended))
    
    def from_index(self, index, query):
        k = self._num_docs  # i.e. return ranking of full corpus
        result = index.find(query["phrase"], n=k, disable_progress=True)
        return self.from_matches(result.matches, query)

    
class NDCGPlotter:
    def __init__(self, gold):
        self._ndcg = NDCGComputer(gold)

        self._gold = gold
        phrase = ([f"mean NDCG"] + self._gold.phrases)[::-1]
        self._phrase = phrase

        p = bokeh.plotting.figure(
            y_range=phrase, plot_width=1000, plot_height=20 * len(self._gold.phrases),
            title="",
            toolbar_location=None, tools="")
        p.x_range = bokeh.models.Range1d(0, 1)
        p.ygrid.visible = False
        p.xaxis.axis_label = 'NDCG'

        self._p = p        
        self._bokeh_handle = None

    def _ndcg_array(self, index):
        ndcg = [self._ndcg.from_index(index, q) for q in self._gold.items]
        return ([np.average(ndcg)] + ndcg)[::-1]
    
    def _format_ndcg(self, ndcg):
        return ['%.1f%%' % (x * 100) for x in ndcg]
            
    def update(self, index):
        ndcg = self._ndcg_array(index)
        
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
            
            
def plot_ndcgs(gold, index):
    plotter = NDCGPlotter(gold)
    plotter.update(index)
    
    
class InteractiveQuery:
    def __init__(self, session, nlp, partition_encoders={}):
        self._session = session
        self._nlp = nlp
        self._partition_encoders = partition_encoders
        self._ui = PartitionMetricWidget(self)
        self._summary_widget = widgets.HTML("")
    
    @property
    def session(self):
        return self._session
    
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
        self._summary_widget.value = html
    
    def create_index(self):
        return self._session.partition("document").index(self._ui.make(), self._nlp)

        
class InteractiveIndexBuilder:
    def __init__(self, session, nlp, partition_encoders={}):
        query_ui = InteractiveQuery(session, nlp, partition_encoders=partition_encoders)        
        query_ui.on_changed()
        self._query_ui = query_ui
        
        tab = widgets.Tab(
            children=[query_ui.summary_widget, query_ui.widget])
        for i, title in enumerate(['Index Summary', 'Edit']):
            tab.set_title(i, title)
        display(tab)
                    
    def build_index(self):
        return self._query_ui.create_index()
    
    
class ResultScoresPlotter:
    def __init__(self, gold, index, query=None):
        self._gold = gold
        self._index = index

        self._doc_formatter = DocFormatter(gold)
        self._ndcg = NDCGComputer(gold)
        
        self._p = None
        self._source = None

        self._result = None
        self._result_html = None
        
        default_query = gold.phrases[0]
        if query is not None:
            candidates = [x for x in gold.phrases if x.startswith(query)]
            if len(candidates) > 0:
                default_query = candidates[0]
        self._default_query = default_query

        self._query_select = bokeh.models.Select(
            title='', options=self._gold.phrases, value=self._default_query)
        
    def _run_query(self):
        query = self._gold.items[self._gold.phrases.index(self._query_select.value)]
        n = 100
        gold_matches = [x["id"] for x in query["matches"]]
        result = self._index.find(query["phrase"], n=n, disable_progress=True)
        self._result = result
            
        ndcg = self._ndcg.from_matches(result.matches, query)
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
            
    def _on_tap(self, event):
        i = math.floor(event.x)
        self._select_rank(i + 1)
        
    def on_update(self):
        qr = self._run_query()
        self._p.title.text = qr["title"]
        self._source.data = qr["data"]
        self._result_html.value = ""
        
    def build(self, rank=None, plot_width=1200, plot_height=250):
        tooltips = """
            @tooltip
        """

        if _has_bokeh_server:
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
        
        if _has_bokeh_server:
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
    
    for drill in drills:
        query = drill.get("query")
        rank = drill.get("rank")
        plotter = ResultScoresPlotter(gold, index, query)
        bk, jp = plotter.build(rank, plot_width=plot_width // len(drills), plot_height=plot_height)
        bks.append(bk)
        jps.append(jp)
        
    bk_root = bokeh.layouts.row(*bks)
    
    if _has_bokeh_server:
        bokeh.io.show(lambda doc: doc.add_root(bk_root))
    else:
        bokeh.io.show(bk_root)

    result_widgets = [widgets.HBox(jps, layout=widgets.Layout(width=f'{plot_width}px'))] if len(jps) > 1 else jps[:1]
    if not _has_bokeh_server:
        result_widgets.append(make_limited_function_warning_widget(
            "change the selected query and the selected rank"))
        
    display(widgets.VBox(result_widgets))

        
def plot_gold(gold):
    G = nx.Graph()

    color = {}
    subset = {}
    
    phrase = {}
    context = {}
    
    formatter = ContextFormatter()
    palette = bokeh.palettes.Spectral4

    doc_template = string.Template("""
        <div style="margin-left:2em">
            <span style="font-variant:small-caps; font-size: 14pt;">${title}</style>
            <span style="float:right; font-size: 10pt;">query: ${phrase}</span>
            <hr>
            <div style="font-variant:normal; font-size: 10pt;">${text}</div>
        </div>
        """)
    
    for i, record in enumerate(gold.items):
        G.add_node(record["phrase"])
        color[record["phrase"]] = palette[0]
        subset[record["phrase"]] = i
        
        phrase_html = f'<i>{record["phrase"]}</i>'
        phrase[record["phrase"]] = phrase_html
        context[record["phrase"]] = ""

        for m in record["matches"]:
            phrase[m["id"]] = ""  # phrase_html + "<hr>"
            context[m["id"]] = doc_template.substitute(
                title=m["work"],
                phrase=record["phrase"],
                text=formatter.format_context(m))

            G.add_edge(record["phrase"], m["id"])
            color[m["id"]] = palette[1]
            subset[m["id"]] = i

    nx.set_node_attributes(G, color, "node_color")
    nx.set_node_attributes(G, subset, "subset")
    nx.set_node_attributes(G, phrase, "phrase")
    nx.set_node_attributes(G, context, "context")
            
    plot = bokeh.models.Plot(
        plot_width=1000, plot_height=400,
        x_range=bokeh.models.Range1d(-1.1, 1.1), y_range=bokeh.models.Range1d(-0.8, 0.8))
    
    node_hover_tool = bokeh.models.HoverTool(
        tooltips="""
        @phrase
        @context
        """)
    plot.add_tools(node_hover_tool)

    graph_renderer = bokeh.plotting.from_networkx(G, nx.multipartite_layout, scale=1, center=(0, 0))
    graph_renderer.node_renderer.glyph = bokeh.models.Circle(size=12, fill_color="node_color")
    graph_renderer.edge_renderer.glyph = bokeh.models.MultiLine(line_color="black", line_alpha=1, line_width=1.5)
    plot.renderers.append(graph_renderer)

    '''
    token_labels = bokeh.models.LabelSet(x='x', y='y', text='token',
        x_offset=5, y_offset=5, source=source['tokens'],
        render_mode='canvas', text_font_size='6pt')
    tok_emb_p.add_layout(token_labels)
    '''
    
    bokeh.io.show(plot)
    