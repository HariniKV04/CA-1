import re
import sys
import os
from collections import defaultdict, Counter


class InvertedIndex:
    def __init__(self, documents):
        self.documents = documents 
        self.index = defaultdict(set)
        self.stopwords = set()

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = text.split()
        return tokens

    def build_index(self):
        word_freq = Counter()
        for sent_id, text in self.documents.items():
            words = self.preprocess(text)
            word_freq.update(words)

        self.stopwords = {w for w, _ in word_freq.most_common(10)}

        for sent_id, text in self.documents.items():
            words = self.preprocess(text)
            for word in set(words):
                if word not in self.stopwords:
                    self.index[word].add(sent_id)

    def index_size(self):
        return sys.getsizeof(self.index)

    def search(self, query):
        tokens = query.upper().split()
        result = None
        operator = None

        def get_postings(term):
            return self.index.get(term.lower(), set())

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in {"AND", "OR", "NOT"}:
                operator = token
            else:
                postings = get_postings(token)
                if result is None:
                    result = postings
                else:
                    if operator == "AND":
                        result = result & postings
                    elif operator == "OR":
                        result = result | postings
                    elif operator == "NOT":
                        result = result - postings
            i += 1

        return result if result else set()



def load_sentences_from_files(folder_path):
    sentence_docs = {}
    sentence_id = 1

    for i in range(1, 5):
        file_path = os.path.join(folder_path, f"doc{i}.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
                for sent in sentences:
                    sentence_docs[sentence_id] = sent
                    sentence_id += 1
        else:
            print(f"File not found: {file_path}")
    return sentence_docs


folder = "D:\SEMESTER 9\IR\Problem Sheets"
sentence_docs = load_sentences_from_files(folder)

engine = InvertedIndex(sentence_docs)
engine.build_index()

print("Stopwords (10 most frequent):", engine.stopwords)
print("Index size (bytes):", engine.index_size())

queries = [
    "retrieval AND information",
    "boolean OR systems",
    "retrieval AND NOT engines"
]

for q in queries:
    result_ids = engine.search(q)
    print(f"\nQuery: {q} → Sentences {result_ids}")
    for sid in result_ids:
        print(f"Sentence {sid}: {sentence_docs[sid]}")
