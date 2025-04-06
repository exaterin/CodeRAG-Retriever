import json
from rag_pipeline import query_repository

def evaluate(queries_file: str):
    with open(queries_file) as f:
        queries = json.load(f)

    total, hits = 0, 0

    for q in queries:
        question = q["question"]
        gold = set(q.get("files") or q.get("answers") or [])
        pred = set(query_repository(question))

        if gold & pred:
            hits += 1

        else:
            print("Question:", question)
            print("Expected:", gold)
            print("Predicted:", pred)
            print("---")

        total += 1

    print(f"Recall@10: {hits / total:.6f}")