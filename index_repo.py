import os
import shutil
import json
from pathlib import Path
from git import Repo
from sentence_transformers import SentenceTransformer
import torch
import faiss
from config import EMBEDDING_MODEL, EMBEDDINGS_FILE, CHUNKS_FILE, INDEX_FILE

model = SentenceTransformer(EMBEDDING_MODEL)


def clone_repo(repo_url: str, clone_path: str = "data/escrcpy"):
    if os.path.exists(clone_path) and os.path.isdir(os.path.join(clone_path, ".git")):
        print("Repository already exists. Skipping clone.")
        return
    print("Cloning repository...")
    import shutil
    shutil.rmtree(clone_path, ignore_errors=True)
    Repo.clone_from(repo_url, clone_path)

def get_code_files(repo_path: str):
    extensions = [".js", ".ts", ".vue", ".json", ".md", ".yml", ".CN", ".1", "readme", "license"]
    files = [p for ext in extensions for p in Path(repo_path).rglob(f"*{ext}")]
    return files

# Splits the file into chunks of chunk_size number of lines
def chunk_code(code: str, chunk_size: int = 15, stride: int = 10):
    lines = code.split("\n")
    return ["\n".join(lines[i:i + chunk_size]) for i in range(0, len(lines) - chunk_size + 1, stride)]


def index_repository():
    print("Indexing...")
    os.makedirs("index", exist_ok=True)
    chunks = []
    metadata = []

    code_files = get_code_files("data/escrcpy")
    print(f"Found {len(code_files)} code files.")

    for file_path in code_files:
        try:
            content = Path(file_path).read_text(encoding='utf-8')
            file_chunks = chunk_code(content)
            for chunk in file_chunks:
                if chunk.strip():
                    chunks.append(chunk)
                    relative_path = str(file_path).replace("data/escrcpy/", "")
                    metadata.append(relative_path)
        except Exception as e:
            print(f"Could not read {file_path}: {e}")
            continue

    print(f"Total chunks: {len(chunks)}")
    if not chunks:
        raise ValueError("No code chunks found to embed.")

    embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)

    # Move to CPU and convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    faiss.normalize_L2(embeddings_np)

    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)

    # Save
    torch.save(embeddings, EMBEDDINGS_FILE)
    with open(CHUNKS_FILE, 'w') as f:
        json.dump(metadata, f)
    faiss.write_index(index, INDEX_FILE)

    print("Indexing complete.")