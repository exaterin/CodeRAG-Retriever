from index_repo import clone_repo, index_repository
from evaluate import evaluate
import os

def run_pipeline(repo_url: str, eval_file: str):
    print("Cloning repository...")
    clone_repo(repo_url)

    print("Indexing codebase...")
    index_repository()

    print("Evaluating retrieval performance...")
    evaluate(eval_file)

if __name__ == "__main__":
    repo_url = "https://github.com/viarotel-org/escrcpy"
    eval_file = "escrcpy-commits-generated.json"

    if not os.path.exists(eval_file):
        print(f"Evaluation file not found: {eval_file}")
    else:
        run_pipeline(repo_url, eval_file)