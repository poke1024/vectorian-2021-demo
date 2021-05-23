import json
from collections import namedtuple


Source = namedtuple("Source", ["work", "author"])


class Pattern:
    def __init__(self, phrase, source):
        self._phrase = phrase
        self._source = source
        self._occurrences = []
    
    @property
    def phrase(self):
        return self._phrase

    @property
    def occurrences(self):
        return self._occurrences
    
    def add_occurrence(self, occ):
        self._occurrences.append(occ)
        occ.attach(self)        
    
    
Evidence = namedtuple("Text", ["context", "phrase"])
    

class Occurrence:
    def __init__(self, unique_id, evidence, source):
        self._unique_id = unique_id
        self._evidence = evidence
        self._source = source
        self._pattern = None
        
    @property
    def unique_id(self):
        return self._unique_id

    @property
    def evidence(self):
        return self._evidence

    @property
    def source(self):
        return self._source

    @property
    def pattern(self):
        return self._pattern
    
    def attach(self, pattern):
        assert self._pattern is None
        self._pattern = pattern
    

class Data:
    def __init__(self, path):
        self._patterns = []
        self._occurrences = []

        with open(path, "r") as f:
            data = json.loads(f.read())

            for entry in data:
                pattern = Pattern(entry["phrase"], entry["source"])
                self._patterns.append(pattern)

                for m in entry["matches"]:
                    occ = Occurrence(
                        m["id"],
                        Evidence(
                            m["context"],
                            m["quote"]),
                        Source(
                            m["work"],
                            m["author"]
                        ))
                    pattern.add_occurrence(occ)
                    self._occurrences.append(occ)
        
    @property
    def patterns(self):
        return self._patterns
    
    @property
    def occurrences(self):
        return self._occurrences
