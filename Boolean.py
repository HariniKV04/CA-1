import os
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")
nltk.download("punkt")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

sentence_docs = {}
sentence_id = 1

folder = r"D:\SEMESTER 9\IR\Problem Sheets"  # raw string to avoid escape issues

for i in range(1, 5):
    file_path = os.path.join(folder, f"doc{i}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
            for sentence in sentences:
                sentence_docs[sentence_id] = sentence
                sentence_id += 1
    else:
        print(f"File not found: {file_path}")


def build_inverted_index(docs):
    inverted = defaultdict(set)
    for doc_id, text in docs.items():
        terms = preprocess(text)  # preprocess each sentence
        for term in set(terms):
            inverted[term].add(doc_id)
    return inverted

inverted = build_inverted_index(sentence_docs)


def boolean_search(query, inverted):
    tokens = query.upper().split()
    result = None
    operator = None

    def get_postings(term):
        return inverted.get(term.lower(), set())

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

queries = [
    "retrieval AND index",
    "data AND mining",
    "boolean AND queries",
    "index OR databases",
    "retrieval AND NOT boolean",
    "boolean OR mining AND NOT index"
]

for q in queries:
    result_ids = boolean_search(q, inverted)
    print(f"\nQuery: {q} → Sentences {result_ids}")
    for sid in result_ids:
        print(f"Sentence {sid}: {sentence_docs[sid]}")
