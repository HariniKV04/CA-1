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
