import string
import collections
import numpy as np
import math
import sklearn.metrics
import itertools
import ipywidgets as widgets

import bokeh.plotting
import bokeh.models
import bokeh.transform
import bokeh.palettes
import bokeh.layouts

from functools import partial
from cached_property import cached_property
from openTSNE import TSNE

from vectorian.embeddings import TokenAveragingSpanEncoder, prepare_docs
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
    def __init__(self, session, nlp, gold):
        self._session = session
        self._nlp = nlp
        self._gold = gold

        self._doc_formatter = DocFormatter(gold)
        self.partition = session.partition("document")

        self.encoders = dict()
        for k, embedding in session.embeddings.items():
            self.encoders[format_embedding_name(k) + " (averaged)"] = CachedPartitionEncoder(
                TokenAveragingSpanEncoder(embedding.factory))    
 
        self._tooltips = """
            <span style="font-variant:small-caps">@work</span>
            <span style="float:right;">"@query" (@similarity%)</span>
            <br>
            <hr>
            @context
            """
        
    def plot(self, embedding, intruder, show_legend=False):
        encoder = self.encoders[embedding]
        intruder_doc = DummyIndex(self.partition).make_query(intruder)
        
        id_to_doc = dict((doc.unique_id, doc) for doc in self._session.documents)
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
        np.nan_to_num(data['vector'], 0)

        tsne = TSNE(
            perplexity=30,
            metric="cosine",
            n_jobs=2,
            random_state=42)

        v = np.array(data['vector'])
        v /= np.linalg.norm(v, axis=1)[:, np.newaxis]

        similarity = [1]
        for x in v[1:]:
            similarity.append(np.dot(v[0], x))
        similarity = np.array(similarity) * 100
        
        X = tsne.fit(v)

        p = bokeh.plotting.figure(
            plot_width=900, plot_height=len(self._gold.phrases) * 30,
            title=f"Sentence Embeddings",
            toolbar_location="below", tools="pan", tooltips=self._tooltips)

        source = bokeh.models.ColumnDataSource({
            'x': X[1:, 0],
            'y': X[1:, 1],
            'work': data["work"][1:],
            'query': data["query"][1:],
            'context': data["context"][1:],
            'similarity': similarity[1:]
        })

        p.circle(
            source=source, size=10, legend_field='query',
            color=bokeh.transform.factor_cmap(
                'query',
                palette=bokeh.palettes.Category20[len(self._gold.phrases)],
                factors=self._gold.phrases),
            alpha=0.8)
        
        p.circle_cross(
            source=bokeh.models.ColumnDataSource({
                'x': X[:1, 0],
                'y': X[:1, 1],
                'work': data["work"][:1],
                'query': data["query"][:1],
                'context': data["context"][:1],
                'similarity': similarity[:1]
            }),
            size=20,
            color="blue",
            line_color="darkblue",
            fill_alpha=0.25)

        if show_legend:
            p.legend.orientation = "vertical"
            p.legend.location = "right"
            p.legend.visible = show_legend
        else:
            p.legend.items = []

        bokeh.io.show(p)


def plot_token_similarity(session, doc, token_sim, ref_token):
    partition = session.partition("document")
    
    color_mapper = bokeh.models.LinearColorMapper(
        palette="Cividis256", low=0, high=1)
    
    sim = partial(session.similarity, token_sim)

    data = collections.defaultdict(list)
    seen = set()
    
    for span in doc.spans(partition):
        for k, token in enumerate(span):
            if token.text not in seen:
                s = max(0, sim(token.text, ref_token))
                data['token'].append(token.text)
                data['sim'].append(s)
                seen.add(token.text)
                    
    data['sim'] = np.array(data['sim'])
    order = np.argsort(data['sim'])[::-1]
    data['token'] = [data['token'][i] for i in order]
    data['sim'] = data['sim'][order]
 
    p = bokeh.plotting.figure(
        y_range=list(reversed(data["token"])), plot_height=len(data['token']) * 20,
        title=f"Token Similarity for {doc.metadata['title']}",
        toolbar_location=None, tools="")

    p.hbar(
        "token", right="sim",
        source=bokeh.models.ColumnDataSource(data), height=0.5,
        color={'field': 'sim', 'transform': color_mapper})
    
    p.x_range = bokeh.models.Range1d(0, 1)
    p.ygrid.grid_line_color = None

    bokeh.io.show(p)

    
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

    def from_matches(self, matches, query):
        recommended = [m.doc.unique_id for m in matches]
        relevant = [m["id"] for m in query["matches"]]
        return ndcg(recommended, relevant, len(recommended))
    
    def from_index(self, index, query):
        k = len(query["matches"])
        result = index.find(query["phrase"], n=k, disable_progress=True)
        return self.from_matches(result.matches, query)

    
class NDCGPlotter:
    def __init__(self, gold):
        self._ndcg = NDCGComputer(gold)

        self._gold = gold
        phrase = ([f"mean NDCG"] + self._gold.phrases)[::-1]
        self._phrase = phrase

        p = bokeh.plotting.figure(
            y_range=phrase, plot_width=800, plot_height=15 * len(self._gold.phrases),
            title=f"NDCG@k, with k=query size",
            toolbar_location=None, tools="")
        p.x_range = bokeh.models.Range1d(0, 1)

        self._p = p        
        self._bokeh_handle = None

    @property
    def widget(self):
        return None
    
    def set_on_change(self, f):
        pass
        
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
                'ndcg_str': self._format_ndcg(ndcg)
            })

            self._p.hbar(
                'phrase', right='ndcg',
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
    
    
class InteractiveQuery:
    def __init__(self, session, nlp, plotter, partition_encoders={}):
        self._session = session
        self._nlp = nlp
        self._plotter = plotter
        self._partition_encoders = partition_encoders
        self._ui = PartitionMetricWidget(self)
    
    @property
    def session(self):
        return self._session
    
    @property
    def partition_encoders(self):
        return self._partition_encoders
    
    @property
    def widget(self):
        return self._ui.widget
    
    def on_changed(self):
        pass
    
    def update_plot(self):
        self._plotter.update(
            self._session.partition("document").index(self._ui.make(), self._nlp))

        
def interact_plot(session, nlp, plotter, partition_encoders={}):
    query_ui = InteractiveQuery(session, nlp, plotter, partition_encoders=partition_encoders)
    
    toggle = widgets.ToggleButtons(
        options=['Result', 'Settings'],
        description='',
        disabled=False,
        button_style='')
    
    def on_toggle(changed):
        query_ui.widget.layout.display = "block" if toggle.value == "Settings" else "none"
        
    toggle.observe(on_toggle, 'value')
    on_toggle(None)
    
    query_ui.update_plot()
        
    recompute_button = widgets.Button(
        description='Recompute',
        button_style='success')
    
    def on_recompute(*args, **kwargs):
        recompute_button.disabled = True
        query_ui.update_plot()
        recompute_button.disabled = False
    
    recompute_button.on_click(on_recompute)
    
    plot_widgets = [x for x in [plotter.widget] if x is not None]
    plotter.set_on_change(on_recompute)
    
    display(widgets.VBox(plot_widgets + [
        widgets.HBox([toggle, recompute_button]),
        query_ui.widget
    ], layout= widgets.Layout(
        display='flex',
        flex_flow='column',
        align_items='center',
        width='800px')))
    
    
class ResultScoresPlotter:
    def __init__(self, gold):
        self._gold = gold
        self._bokeh_handle = None

        self._doc_formatter = DocFormatter(gold)
        self._ndcg = NDCGComputer(gold)

        self._query_index_widget = widgets.IntSlider(
            description='query', min=1, max=len(gold.phrases))
        
    @property
    def widget(self):
        return self._query_index_widget

    def set_on_change(self, f):
        self._query_index_widget.observe(f, 'value')
    
    def update(self, index):
        query = self._gold.items[self._query_index_widget.value - 1]
        n = 100
        gold_matches = [x["id"] for x in query["matches"]]
        result = index.find(query["phrase"], n=n, disable_progress=True)
        
        tooltips = """
            @tooltip
        """
        
        def update_source(source):
            source['rank'] = [str(i) for i in range(1, n + 1)]
            source['score'] = [m.score for m in result.matches]
            source['color'] = [("green" if m.doc.unique_id in gold_matches else "red") for m in result.matches]
            source['tooltip'] = [self._doc_formatter(m.prepared_doc) for m in result.matches]
            
        ndcg = self._ndcg.from_matches(result.matches, query)
        title = f"Scores for Query '{query['phrase']}', NDCG={ndcg * 100:.1f}%"

        if self._bokeh_handle is None:
            data = {}
            update_source(data)

            p = bokeh.plotting.figure(
                x_range=data['rank'], plot_width=1000, plot_height=250,
                title=title,
                toolbar_location=None, tools="", tooltips=tooltips)
            self._p = p
            
            source = bokeh.models.ColumnDataSource(data)
            self._source = source

            hbar = p.vbar(
                "rank", top="score", color="color", source=source)

            #p.y_range = bokeh.models.Range1d(0, 1)
            p.xaxis.major_label_orientation = np.pi / 2

            self._bokeh_handle = bokeh.io.show(p, notebook_handle=True)
        else:
            self._p.title.text = title
            update_source(self._source.data)
            bokeh.io.push_notebook(handle=self._bokeh_handle)
