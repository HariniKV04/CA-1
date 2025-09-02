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
    
    

