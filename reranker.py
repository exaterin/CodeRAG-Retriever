from sentence_transformers import CrossEncoder


rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_candidates(query: str, candidates: list[str]) -> list[str]:
    pairs = [(query, c) for c in candidates]
    scores = rerank_model.predict(pairs)
    return [x for _, x in sorted(zip(scores, candidates), key=lambda pair: -pair[0])]