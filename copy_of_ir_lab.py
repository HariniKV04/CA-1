import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from collections import defaultdict

# Sample documents
docs = [
    "information retrieval uses binary independence model",
    "probabilistic models are important in information retrieval",
    "bm25 is derived from the binary independence model",
    "neural networks are widely used for information retrieval"
]

# Queries
query = "binary independence model"

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# =========================
# Phase 1: BIM without relevance
# =========================
def BIM_no_rel(query, docs):
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(docs).toarray()
    terms = vectorizer.get_feature_names_out()
    q_vec = vectorizer.transform([query]).toarray()[0]

    N = len(docs)
    df = np.sum(X, axis=0)  # document frequency of each term

    # Phase 1 assumptions
    p = 0.5
    q = (df + 0.5) / (N + 1)

    # RSJ weights
    weights = np.log((p * (1 - q)) / (q * (1 - p)))

    # Score documents
    scores = X @ (q_vec * weights)
    ranking = np.argsort(-scores)

    print("Query:", query)
    print("\nDocument Ranking (Phase 1 - No Relevance Feedback):")
    for i in ranking:
        print(f"Doc {i}: '{docs[i]}' | Score = {scores[i]:.4f}")


# =========================
# Phase 2: BIM with relevance
# =========================
def BIM_with_rel(query, docs, relevant_docs):
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(docs).toarray()
    terms = vectorizer.get_feature_names_out()
    q_vec = vectorizer.transform([query]).toarray()[0]

    # Partition docs into relevant and non-relevant
    non_relevant_docs = [i for i in range(len(docs)) if i not in relevant_docs]
    R = len(relevant_docs)
    NR = len(non_relevant_docs)

    # Document frequencies in relevant and non-relevant sets
    df_rel = np.sum(X[relevant_docs], axis=0) if R > 0 else np.zeros(len(terms))
    df_nonrel = np.sum(X[non_relevant_docs], axis=0) if NR > 0 else np.zeros(len(terms))

    # Probabilities
    p = (df_rel + 0.5) / (R + 1) if R > 0 else np.full(len(terms), 0.5)
    q = (df_nonrel + 0.5) / (NR + 1) if NR > 0 else np.full(len(terms), 0.5)

    # RSJ weights
    weights = np.log((p * (1 - q)) / (q * (1 - p)))

    # Score documents
    scores = X @ (q_vec * weights)
    ranking = np.argsort(-scores)

    print("Query:", query)
    print("\nDocument Ranking (Phase 2 - With Relevance Feedback):")
    for i in ranking:
        print(f"Doc {i}: '{docs[i]}' | Score = {scores[i]:.4f}")

# Phase 1
BIM_no_rel(query, docs)

print("\n" + "="*60 + "\n")

# Phase 2 (assume docs 0 and 2 are relevant)
BIM_with_rel(query, docs, relevant_docs=[0, 2])

#VSM
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def VSM_tfidf(query,docs):
    # Step 1: Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)   # Document-term matrix
    q_vec = vectorizer.transform([query])  # Query vector

    # Step 2: Compute cosine similarity between query and each document
    scores = cosine_similarity(q_vec, X).flatten()

    # Step 3: Rank documents
    ranking = np.argsort(-scores)

    print("Query:", query)
    print("\nDocument Ranking (Vector Space Model):")
    for i in ranking:
        print(f"Doc {i}: '{docs[i]}' | Score = {scores[i]:.4f}")

#VSM with weighted tf
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def VSM_weighted_tf(query, docs):
    # Parameters
    alpha = 0.85   # similarity threshold
    N = len(docs)  # number of documents

    # Step 1: Preprocess documents + query into tokens
    tokenized_docs = [doc.lower().split() for doc in docs]
    query_tokens = query.lower().split()

    # Step 2: Build vocabulary
    vocab = sorted(set(term for doc in tokenized_docs for term in doc) | set(query_tokens))
    vocab_index = {term: idx for idx, term in enumerate(vocab)}

    # Step 3: Compute df (document frequency for each term)
    df = {term: sum(1 for doc in tokenized_docs if term in doc) for term in vocab}

    # Step 4: Build weighted document-term matrix
    X = np.zeros((N, len(vocab)))

    for i, doc in enumerate(tokenized_docs):
        len_i = len(doc)
        tf_counts = Counter(doc)
        for term, tf in tf_counts.items():
            j = vocab_index[term]
            # Apply given formula
            tf_norm = tf / len_i
            idf = math.log((N + 1) / (0.5 + df[term]))
            X[i, j] = tf_norm * idf

    # Step 5: Build query vector with same formula
    q_vec = np.zeros(len(vocab))
    len_q = len(query_tokens)
    tf_counts_q = Counter(query_tokens)
    for term, tf in tf_counts_q.items():
        if term in vocab_index:
            j = vocab_index[term]
            tf_norm = tf / len_q
            idf = math.log((N + 1) / (0.5 + df[term]))
            q_vec[j] = tf_norm * idf

    # Step 6A: Query similarity (ranking)
    scores = cosine_similarity([q_vec], X).flatten()
    ranking = np.argsort(-scores)

    # Step 6B: Duplicate detection among documents
    cos_sim = cosine_similarity(X)
    duplicates = []
    for i in range(N):
        for j in range(i + 1, N):
            if cos_sim[i, j] > alpha:
                duplicates.append((i, j, cos_sim[i, j]))

    # --- Results ---
    print("Query:", query, "\n")

    print("Document Ranking (Custom VSM):")
    for i in ranking:
        print(f"Doc {i}: '{docs[i]}' | Score = {scores[i]:.4f}")

    print("\nCosine Similarity Matrix (Docs vs Docs):\n", cos_sim, "\n")

    if duplicates:
        print(f"Documents detected as duplicates (α = {alpha}):")
        for (i, j, sim) in duplicates:
            print(f"Doc {i} ↔ Doc {j} | Similarity = {sim:.4f}")
    else:
        print("No duplicates found above threshold.")

print("\nVSM model")
VSM_tfidf(query,docs)

print("\nVSM model with weighted tf")
VSM_weighted_tf(query,docs)

#inverted index
from collections import defaultdict

def build_inverted_index(docs):
    inverted_index = defaultdict(list)

    for doc_id, doc in enumerate(docs):
        terms = doc.lower().split()
        for term in set(terms):  # use set to avoid duplicate entries for same doc
            inverted_index[term].append(doc_id)

    return dict(inverted_index)

# Build index
inv_index = build_inverted_index(docs)

# Print inverted index
for term, doc_list in inv_index.items():
    print(f"{term}: {doc_list}")

#Boolean model using inverted index

import re
from collections import defaultdict

docs = [
    "information retrieval uses binary independence model",
    "probabilistic models are important in information retrieval",
    "bm25 is derived from the binary independence model",
    "neural networks are widely used for information retrieval"
]

# Step 1: Build Inverted Index
def build_inverted_index(docs):
    inverted_index = defaultdict(set)
    for doc_id, doc in enumerate(docs):
        terms = doc.lower().split()
        for term in set(terms):  # avoid duplicates
            inverted_index[term].add(doc_id)
    return dict(inverted_index)

inv_index = build_inverted_index(docs)
print("Inverted Index: ")
# Print inverted index
for term, doc_list in inv_index.items():
    print(f"{term}: {doc_list}")

# Step 2: Boolean Query Evaluation
def parse_boolean_query(query, inv_index, total_docs):

    q = query.lower()

    # Replace operators with Python equivalents and handle NOT
    q = q.replace(" and ", " & ")
    q = q.replace(" or ", " | ")
    q = q.replace(" not ", " ~")

    # Ensure terms are replaced as set(inv_index[term])
    for term in set(re.findall(r'\w+', q)):
      if term not in ["&", "|", "~", "(", ")"]: # Avoid replacing operators
        if term in inv_index:
            q = re.sub(rf'\b{term}\b', f'set({inv_index[term]})', q)
        else:
            q = re.sub(rf'\b{term}\b', 'set()', q) # unknown term -> empty set

    q = re.sub(r'~set\(', 'all_docs - set(', q)

    # Universal set of all documents
    all_docs = set(range(total_docs))
    allowed_globals = {"__builtins__": None}
    allowed_locals = {"set": set, "all_docs": all_docs}

    # Evaluate expression
    try:
        result = eval(q, allowed_globals, allowed_locals)
        # Ensure the result is a set
        if isinstance(result, set):
            return result
        else:
            # If eval result is not a set, it indicates a parsing error
            print(f"Warning: Query evaluation resulted in non-set type: {type(result)}")
            return set()
    except Exception as e:
        print(f"Error evaluating query '{query}': {e}")
        return set()

# -------------------------
queries = [
    "retrieval AND binary OR neural NOT probabilistic",
    "(binary AND retrieval) OR (neural AND NOT probabilistic)",
    "neural AND (retrieval OR model)"
]

for query in queries:
    matching_docs = parse_boolean_query(query, inv_index, len(docs))
    print("\nQuery:", query)
    if matching_docs:
        for doc_id in sorted(matching_docs):
            print(f"  Doc {doc_id}: {docs[doc_id]}")
    else:
        print("  No documents matched")

#reading doc
import os

# Path to the folder containing your documents
folder_path = "C:/Users/ASHERE JESWIN/Downloads/IR_DOC"

docs = {}

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = " ".join(line.strip() for line in f if line.strip())
            docs[filename] = content

# Example output
for name, text in docs.items():
    print(f"{name}: {text}\n")


# Print the docs list
print("Sample documents")
print("docs = [")
for d in docs.values():
    print(f'    "{d}",')
print("]")
