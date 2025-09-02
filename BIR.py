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
