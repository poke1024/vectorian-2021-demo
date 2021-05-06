import string
import collections
import numpy as np
import math
import sklearn.metrics
import itertools
import functools
import ipywidgets as widgets
import requests
import markdown

import bokeh.plotting
import bokeh.models
import bokeh.transform
import bokeh.palettes
import bokeh.layouts

from functools import partial
from cached_property import cached_property
from openTSNE import TSNE
from IPython.core.display import HTML, display

from vectorian.embeddings import TokenEmbeddingAggregator, prepare_docs
from vectorian.embeddings import CachedPartitionEncoder
from vectorian.index import DummyIndex
from vectorian.metrics import TokenSimilarity, CosineSimilarity
from vectorian.interact import PartitionMetricWidget
    

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
    
    @cached_property
    def doc_digest_to_ids(self):
        digests = {}
        for query in self._data:
            for m in query["matches"]:
                digests[f"{m['work']}: {m['context']}"[:80] + "..."] = m['id']
        return digests
        

    
class DocFormatter:
    def __init__(self, gold):
        self._template = string.Template("""
            <div style="margin-left:2em">
                <span style="font-variant:small-caps; font-size: 14pt;">${title}</style>
                <span style="float:right; font-size: 10pt;">query: ${phrase}</span>
                <hr>
                <div style="font-variant:normal; font-size: 10pt;">${text}</div>
            </div>
            """)
        self._gold = gold
    
    def enhanced_doc_text(self, doc):
        with doc.text() as text_ref:
            text = text_ref.get()
        quote = self._gold.by_id[doc.unique_id]["match"]["quote"]
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
        
    def __call__(self, doc):
        return self._template.substitute(
            phrase=self._gold.by_id[doc.unique_id]["query"]["phrase"],
            title=doc.metadata["title"],
            text=self.enhanced_doc_text(doc))


def format_embedding_name(name):
    if name.startswith("https://github.com"):
        return "/".join(name.split("/")[4:])
    else:
        return name


class EmbeddingPlotter:
    def __init__(self, session, nlp, gold, aggregator):
        self._session = session
        self._nlp = nlp
        self._gold = gold

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
    
        self._doc_tsne = TSNE(
            perplexity=30,
            metric="cosine",
            n_jobs=2,
            random_state=42)

        self._tok_tsne = TSNE(
            perplexity=50,  # 10
            metric="cosine",
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
                contexts.append(self._doc_formatter.enhanced_doc_text(doc))

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
        
    def mk_plot(self, bokeh_doc):
        embedding_select = bokeh.models.Select(
            title="",
            value=sorted(self.encoders.keys())[0],
            options=sorted(self.encoders.keys()),
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
        

        source = dict((k, bokeh.models.ColumnDataSource(v)) for k, v in self._compute_source_data(
            embedding_select.value, "").items())
        
        cmap = bokeh.transform.factor_cmap(
            'query',
            palette=bokeh.palettes.Category20[len(self._gold.phrases)],
            factors=self._gold.phrases)

        
        tok_emb_p = bokeh.plotting.figure(
            plot_width=400, plot_height=600,
            title=f"Token Embeddings",
            toolbar_location="right",
            tools="pan,wheel_zoom,box_zoom,reset",
            tooltips=self._tok_emb_tooltips,
            visible=False)
        
        tok_emb_p.circle(
            source=source['tokens'],
            size=10,
            #legend_field='query',
            color=cmap,
            alpha=0.8)
        
        tok_emb_status = bokeh.models.Div(text="")
        
        token_labels = bokeh.models.LabelSet(x='x', y='y', text='token',
            x_offset=5, y_offset=5, source=source['tokens'],
            render_mode='canvas', text_font_size='6pt')
        tok_emb_p.add_layout(token_labels)
        
        
        doc_emb_p = bokeh.plotting.figure(
            plot_width=600, plot_height=600,
            title=f"Document Embeddings",
            toolbar_location="left",
            tools="pan, lasso_select, box_select",
            active_drag="lasso_select",
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
        

        def set_tok_emb_status(text):
            tok_emb_status.text = f"""<p style="width:100%; font-weight: bold; text-align:center;">{text}</p>"""

        def update_token_plot(max_token_count=750):
            embedding = self.encoders[embedding_select.value].embedding
            if embedding is None:
                clear_token_plot()
                return

            selected = source['docs'].selected.indices
            if not selected:
                clear_token_plot()
                return
            
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

            tok_emb_p.visible = True
            tok_emb_status.visible = False
            
        def toggle_legend(attr, old, new):
            if 0 in options_cb.active:
                legend.visible = True
            else:
                legend.visible = False
                
        def update_document_embedding_plot(attr, old, new):
            if query_tabs.active == 0:
                intruder = ""
            else:
                intruder = [intruder_select, intruder_free][query_tabs.active - 1].value
            for k, v in self._compute_source_data(
                embedding_select.value, intruder).items():
                source[k].data = v
            update_token_plot()
                
        def clear_token_plot():
            source['tokens'].data = self._empty_token_data
            tok_emb_p.visible = False
            tok_emb_status.visible = True
            tok_emb_status.text = ""
            
        def trigger_token_plot_update(tok_plot_state):
            if self._tok_plot_state == tok_plot_state:
                update_token_plot()

        def selection_change(attr, old, new):
            clear_token_plot()
            set_tok_emb_status("Computing. Please Wait...")
            self._tok_plot_state += 1
            bokeh_doc.add_timeout_callback(functools.partial(
                trigger_token_plot_update, self._tok_plot_state), 500)
                        
                
        options_cb.visible = False  # broken in bokeh
                
        embedding_select.on_change("value", update_document_embedding_plot)
        intruder_select.on_change("value", update_document_embedding_plot)
        intruder_free.on_change("value", update_document_embedding_plot)
        query_tabs.on_change("active", update_document_embedding_plot)
        source['docs'].selected.on_change('indices', selection_change)
        options_cb.on_change("active", toggle_legend)

        source['docs'].js_on_change("data", bokeh.models.CustomJS(args={'p': doc_emb_p}, code="""
            p.reset.emit();
        """))
        source['tokens'].js_on_change("data", bokeh.models.CustomJS(args={'p': tok_emb_p}, code="""
            p.reset.emit();
        """))
        
        bokeh_doc.add_root(bokeh.layouts.column(
            bokeh.layouts.column(embedding_select, query_tabs, background="#F0F0F0"),
            bokeh.layouts.row(
                doc_emb_p,
                bokeh.layouts.column(tok_emb_status, tok_emb_p)),
            options_cb,
            sizing_mode="stretch_width"))
            

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
        
        doc_id = self._gold.doc_digest_to_ids[self._doc_select.value]
        found = [x for x in self._session.documents if x.unique_id == doc_id]
        doc = found[0]

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
            
    def __init__(self, session, nlp, gold, n_figures=2, top_n=15):
        self._session = session
        self._nlp = nlp
        self._gold = gold
        
        doc_digests = list(gold.doc_digest_to_ids.keys())
        self._embedding_names = sorted(session.embeddings.keys())

        self._figures = None
        self._n_figures = min(n_figures, len(session.embeddings))
        self._height_per_token = 20
        self._top_n = top_n
        
        self._token_text = bokeh.models.TextInput(value="high")
        self._doc_select = bokeh.models.Select(options=doc_digests, value=doc_digests[0])
        #self._top_n = bokeh.models.Slider(start=5, end=100, step=5, value=15, title="top n")

        self._partition = session.partition("document")
        
        self._token_text.on_change("value", lambda attr, old, new: self._update())
        self._doc_select.on_change("value", lambda attr, old, new: self._update())
        #self._top_n.on_change("value", lambda attr, old, new: self._update())

        self._color_mapper = bokeh.models.LinearColorMapper(
            palette="Viridis256", low=0, high=1)
        
    def _create_figure_record(self, index):
        embedding_select = bokeh.models.Select(
            options=self._embedding_names, value=self._embedding_names[index])
        embedding_select.on_change("value", lambda attr, old, new: self._update(index=index))
        
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
        
    def create(self, bokeh_doc):
        self._figures = [self._create_figure_record(i) for i in range(self._n_figures)]
        
        state = self._create_state()

        for figure, data in zip(self._figures, state['data']):
            self._init_figure_record(figure, data)

        p = self._figures[0]['figure']
        p.yaxis.axis_label = state['label']
        for x in self._figures:
            p = x['figure']
            p.xaxis.axis_label = "cosine similarity"
            
        bokeh_doc.add_root(bokeh.layouts.column(
            self._token_text,
            self._doc_select,
            bokeh.layouts.row(*[
                bokeh.layouts.column(x['embedding_select'], x['figure']) for x in self._figures])
        ))

            
def plot_token_similarity(session, nlp, gold, n_figures=2, top_n=15):
    plotter = TokenSimilarityPlotter(session, nlp, gold, n_figures=n_figures, top_n=top_n)
    bokeh.io.show(plotter.create)
    

    
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
        
        self._bokeh_doc = None
        self._p = None
        self._source = None

        self._result = None
        self._result_html = None
        
        default_query = gold.phrases[0]
        if query is not None:
            candidates = [x for x in gold.phrases if x.startswith(query)]
            if len(candidates) > 0:
                default_query = candidates[0]

        self._query_select = bokeh.models.Select(
            title='', options=gold.phrases, value=default_query)
        
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
                'tooltip': [self._doc_formatter(m.prepared_doc) for m in result.matches]
            }
        }
            
    def create_plot(self, index):
        if self._bokeh_doc is None:
            bokeh.io.show(functools.partial(self._create_plot, index))
        else:
            self._create_plot(index, self._bokeh_doc)
            
    def _on_tap(self, event):
        i = math.floor(event.x)
        source = self._source
        
        hue = source.data["base_hue"][:]
        hue[i] = 0.1 if hue[i] < 0.5 else 0.9
        source.data["hue"] = hue
              
        from vectorian.render.excerpt import ExcerptRenderer
        from vectorian.render.render import Renderer
        from vectorian.render.location import LocationFormatter
        
        renderer = Renderer(
            [ExcerptRenderer()],
            LocationFormatter())
        
        html = renderer.to_html([self._result.matches[i]])
        self._result_html.value = html
        
    def on_update(self):
        qr = self._run_query()
        self._p.title.text = qr["title"]
        self._source.data = qr["data"]
        self._result_html.value = ""
        
    def create_plot(self, index, bokeh_doc):
        plot_width = 1200
        tooltips = """
            @tooltip
        """

        qr = self._run_query()

        p = bokeh.plotting.figure(
            x_range=qr['data']['rank'], plot_width=plot_width, plot_height=250,
            title=qr['title'],
            toolbar_location=None, tools="", tooltips=tooltips)
        self._p = p

        p.xaxis.axis_label = 'rank'
        p.yaxis.axis_label = 'NDCG'
        
        p.on_event(bokeh.events.Tap, self._on_tap)
        self._query_select.on_change('value', lambda attr, old, new: self.on_update())
        
        source = bokeh.models.ColumnDataSource(qr["data"])
        self._source = source

        mapper = bokeh.transform.linear_cmap(
            field_name='hue', palette=bokeh.palettes.RdBu6, low=0, high=1)
        vbar = p.vbar(
            "rank", top="score", color=mapper, source=source, width=0.8)

        #p.y_range = bokeh.models.Range1d(0, 1)
        p.xaxis.major_label_orientation = np.pi / 2
        p.xgrid.visible = False

        bokeh_doc.add_root(bokeh.layouts.column(self._query_select, p))
        self._bokeh_doc = bokeh_doc

        self._result_html = widgets.HTML("")
        display(self._result_html)


def plot_results(gold, index, query=None):
    plotter = ResultScoresPlotter(gold, index, query)
    bokeh.io.show(functools.partial(plotter.create_plot, index))
