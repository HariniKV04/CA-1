






























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

"""## Boolean"""

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

"""## Similarity"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


def binary_distance(title1, title2):
    return 0 if title1.strip().lower() == title2.strip().lower() else 1

def is_duplicate_cosine(new_doc, existing_docs, threshold=0.85):
    corpus = existing_docs + [new_doc]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    sim_matrix = cosine_similarity(vectors[-1], vectors[:-1])
    max_sim = sim_matrix.max()
    return max_sim >= threshold, max_sim

def get_k_shingles(text, k=5):
    words = re.findall(r'\w+', text.lower())
    shingles = set()
    for i in range(len(words) - k + 1):
        shingles.add(tuple(words[i:i+k]))
    return shingles

def is_duplicate_jaccard(new_doc, existing_docs, k=5, threshold=0.85):
    new_shingles = get_k_shingles(new_doc, k)
    for i, doc in enumerate(existing_docs):
        doc_shingles = get_k_shingles(doc, k)
        jaccard_sim = len(new_shingles & doc_shingles) / len(new_shingles | doc_shingles)
        if jaccard_sim >= threshold:
            return True, i, jaccard_sim
    return False, None, 0

def is_duplicate_bm25(new_doc, existing_docs, threshold=0.85):
    tokenized_corpus = [re.findall(r'\w+', doc.lower()) for doc in existing_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    new_tokens = re.findall(r'\w+', new_doc.lower())
    scores = bm25.get_scores(new_tokens)
    max_score = max(scores) if scores else 0
    # Normalize score between 0 and 1 (simple approach)
    normalized_score = max_score / max(scores) if max(scores) > 0 else 0
    return normalized_score >= threshold, max_score

def check_plagiarism(new_doc, new_title, database, titles, alpha=0.85, k=5):
    for t in titles:
        if binary_distance(new_title, t) == 0:
            return True, "Duplicate: Title matches exactly"

    is_dup, sim = is_duplicate_cosine(new_doc, database, threshold=alpha)
    if is_dup:
        return True, f"Duplicate: Cosine similarity {sim:.2f} >= {alpha}"

    is_dup, idx, jaccard_sim = is_duplicate_jaccard(new_doc, database, k=k, threshold=alpha)
    if is_dup:
        return True, f"Duplicate: Jaccard similarity {jaccard_sim:.2f} >= {alpha} (doc {idx+1})"

    is_dup, score = is_duplicate_bm25(new_doc, database, threshold=alpha)
    if is_dup:
        return True, f"Duplicate: BM25 similarity {score:.2f} >= {alpha}"

    return False, "Document is unique"

database_titles = ["Information Retrieval Basics", "Inverted Index in Search Engines"]
database_docs = [
    "Information retrieval is the process of obtaining useful information from large collections.",
    "An inverted index maps terms to the documents in which they occur."
]

new_title = "Information Retrieval Basics"
new_content = "Information retrieval involves ranking algorithms and indexing large document collections."

is_dup, msg = check_plagiarism(new_content, new_title, database_docs, database_titles)
print(msg)

"""## VSM"""

import re
import os
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter

# Download resources (only first time)
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


documents = {}
folder = r"D:\SEMESTER 9\IR\Problem Sheets"

for i in range(1, 5):
    folder_path = os.path.join(folder, f"doc{i}.txt")
    with open(folder_path, "r", encoding="utf-8") as f:
        documents[f"doc{i}"] = preprocess(f.read())

vocab = set()
for tokens in documents.values():
    vocab.update(tokens)
vocab = sorted(vocab)

print(vocab)
vocab_index = {term: i for i, term in enumerate(vocab)}

tf = {}
for doc_id, tokens in documents.items():
    counts = Counter(tokens)
    tf[doc_id] = [counts.get(term, 0) for term in vocab]

N = len(documents)
df = defaultdict(int)
for term in vocab:
    for tokens in documents.values():
        if term in tokens:
            df[term] += 1
idf = {term: math.log(N / (df[term] + 1)) for term in vocab}


tfidf = {}
for doc_id, vec in tf.items():
    tfidf[doc_id] = [vec[i] * idf[vocab[i]] for i in range(len(vocab))]

def cosine_similarity(vec1, vec2):
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def jaccard_similarity(vec1, vec2):
    num = sum(min(a,b) for a,b in zip(vec1,vec2))
    den = sum(max(a,b) for a,b in zip(vec1,vec2))
    return num/den if den else 0.0

def dice_similarity(vec1, vec2):
    dot = sum(a*b for a,b in zip(vec1,vec2))
    norm1 = sum(a*a for a in vec1)
    norm2 = sum(b*b for b in vec2)
    return 2*dot / (norm1+norm2) if (norm1+norm2) else 0.0

def inner_product(vec1, vec2):
    return sum(a*b for a,b in zip(vec1,vec2))

def rank_documents(query):
    query_tokens = preprocess(query)
    counts = Counter(query_tokens)
    query_vec = [counts.get(term, 0) * idf.get(term, 0) for term in vocab]

    scores = {"cosine": {}, "jaccard": {}, "dice": {}, "dot": {}}
    for doc_id, doc_vec in tfidf.items():
        scores["cosine"][doc_id] = cosine_similarity(query_vec, doc_vec)
        scores["jaccard"][doc_id] = jaccard_similarity(query_vec, doc_vec)
        scores["dice"][doc_id] = dice_similarity(query_vec, doc_vec)
        scores["dot"][doc_id] = inner_product(query_vec, doc_vec)

    best_docs = {}
    for sim in scores:
        best_docs[sim] = max(scores[sim].items(), key=lambda x: x[1])

    return scores, best_docs

query = "information retrieval system"
scores, best_docs = rank_documents(query)

print("Query:", query)
print("\nBest document per similarity measure:")
for sim, (doc, score) in best_docs.items():
    print(f"{sim.capitalize()} → {doc} (score={score:.4f})")

"""## BIM"""

import os
import re
from collections import defaultdict
import math

def load_documents(folder_path):
    documents = {}
    for i in range(1, 5):
        file_path = os.path.join(folder_path, f"doc{i}.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().lower()
                words = re.findall(r'\w+', text)  # tokenize
                documents[f"doc{i}"] = set(words)  # store as set for BIM
        else:
            print(f"File not found: {file_path}")
    return documents

folder = r"D:\SEMESTER 9\IR\Problem Sheets"
documents = load_documents(folder)


def compute_bim_weights(documents, query):
    N = len(documents)
    query_terms = set(re.findall(r'\w+', query.lower()))

    # Count document frequency
    df = defaultdict(int)
    for doc, terms in documents.items():
        for term in query_terms:
            if term in terms:
                df[term] += 1

    # Compute RSJ / idf-like weights
    weights = {}
    for term in query_terms:
        n_t = df[term]
        weights[term] = math.log((N - n_t + 0.5) / (n_t + 0.5))  # Robertson-Sparck Jones approximation
    return weights

def rank_documents(documents, query):
    weights = compute_bim_weights(documents, query)
    scores = {}

    for doc, terms in documents.items():
        score = 0
        for term, w in weights.items():
            if term in terms:
                score += w
        scores[doc] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

query = "information retrieval systems"
ranked_docs = rank_documents(documents, query)

for doc, score in ranked_docs:
    print(f"{doc}: {score:.4f}")


def evaluate(retrieved_docs, relevant_docs):
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)

    true_positive = len(retrieved_set & relevant_set)
    precision = true_positive / len(retrieved_set) if retrieved_set else 0
    recall = true_positive / len(relevant_set) if relevant_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1

# Example usage
retrieved_docs = ["doc2", "doc4", "doc3"]  # top-k retrieved by BIM or VSM
relevant_docs = ["doc2", "doc4"]           # ground truth

precision, recall, f1 = evaluate(retrieved_docs, relevant_docs)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

"""## BIR"""

import os
import re
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return set(tokens)  # binary model → set of terms


def load_documents(folder, n_files=4):
    documents = {}
    for i in range(1, n_files + 1):
        path = os.path.join(folder, f"doc{i}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                documents[f"doc{i}"] = preprocess(f.read())
        else:
            print(f"File not found: {path}")
    return documents


def compute_phase1_weights(documents):
    N = len(documents)
    df = defaultdict(int)
    for terms in documents.values():
        for term in terms:
            df[term] += 1

    weights = {}
    for term, dk in df.items():
        weights[term] = math.log((N - dk + 0.5) / (dk + 0.5))
    return weights



def compute_phase2_weights(documents, relevant_docs):
    N = len(documents)
    Nr = len(relevant_docs)
    df = defaultdict(int)
    rk = defaultdict(int)

    for doc_id, terms in documents.items():
        for term in terms:
            df[term] += 1
            if doc_id in relevant_docs:
                rk[term] += 1

    weights = {}
    for term in df:
        r = rk.get(term, 0)
        dk = df[term]
        numerator = (r + 0.5) / (Nr - r + 0.5)
        denominator = (dk - r + 0.5) / (N - dk - Nr + r + 0.5)
        weights[term] = math.log(numerator / denominator)
    return weights

def rank_documents_phase(query, documents, weights):
    query_terms = preprocess(query)
    scores = {}
    for doc_id, terms in documents.items():
        score = sum(weights.get(term, 0) for term in query_terms if term in terms)
        scores[doc_id] = score
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

folder = r"D:\SEMESTER 9\IR\Problem Sheets"
documents = load_documents(folder, n_files=4)

query = "information retrieval"
print("Query:", query, "\n")

phase1_weights = compute_phase1_weights(documents)
phase1_scores = rank_documents_phase(query, documents, phase1_weights)
print("Phase I Rankings (No Feedback):")
for doc_id, score in phase1_scores.items():
    print(f"{doc_id}: score={score:.4f}")


relevant_docs = {"doc1"}
phase2_weights = compute_phase2_weights(documents, relevant_docs)
phase2_scores = rank_documents_phase(query, documents, phase2_weights)
print("\nPhase II Rankings (With Feedback):")
for doc_id, score in phase2_scores.items():
    print(f"{doc_id}: score={score:.4f}")
