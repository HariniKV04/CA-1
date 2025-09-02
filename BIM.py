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

