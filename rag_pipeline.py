import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, INDEX_FILE, CHUNKS_FILE, TOP_K
from reranker import rerank_candidates

model = SentenceTransformer(EMBEDDING_MODEL)

def query_repository(question: str):
    question_emb = model.encode(question, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    faiss.normalize_L2(question_emb)
    index = faiss.read_index(INDEX_FILE)
    scores, indices = index.search(question_emb.reshape(1, -1), TOP_K * 3)  # Retrieve more for reranking
    with open(CHUNKS_FILE) as f:
        metadata = json.load(f)
    candidates = [metadata[i] for i in indices[0]]
    return rerank_candidates(question, candidates)[:TOP_K]