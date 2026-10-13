"""
Retrieval over the description document of a dataset.

Convention: a dataset's description lives next to it, with the same name.

    sales.csv        the data
    sales.md         its description (.md or .txt, optional)
    sales.rag/       its index, generated from the description

Usage:

    python rag.py build  sales.csv               chunk + embed + save the index
    python rag.py search sales.csv "a question"  search from the terminal

Rebuild the index every time the description changes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import ollama


EMBED_MODEL = "nomic-embed-text"

# nomic-embed-text works best when each text says what role it plays
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

DESCRIPTION_EXTENSIONS = (".md", ".txt")

# Taille max d'un morceau quand le document n'a pas de titres "## "
MAX_CHUNK_CHARS = 800


# --------------------------------------------------
# 0. Où se trouvent la description et l'index d'un dataset
# --------------------------------------------------

def description_path(csv_path):
    """Return the description file next to the CSV, or None if there is none."""

    base = Path(csv_path).with_suffix("")

    for extension in DESCRIPTION_EXTENSIONS:
        candidate = base.with_suffix(extension)
        if candidate.exists():
            return candidate

    return None


def index_dir(csv_path):
    """sales.csv -> sales.rag/"""

    base = Path(csv_path).with_suffix("")

    return base.parent / (base.name + ".rag")


# --------------------------------------------------
# 1. Découper
# --------------------------------------------------

def split_into_chunks(text):
    """
    Split on '## ' headings when the document has some.
    Otherwise group paragraphs into chunks of at most MAX_CHUNK_CHARS.
    """

    text = "\n" + text.strip()

    # Cas 1 : document structuré avec des titres "## "
    if "\n## " in text:

        chunks = []

        # Le premier bloc est ce qui précède le premier "## " (le titre "# ...")
        for section in text.split("\n## ")[1:]:

            title, _, body = section.partition("\n")

            chunks.append({
                "title": title.strip(),
                "text": f"{title.strip()}\n{body.strip()}",
            })

        return chunks

    # Cas 2 : texte libre, on regroupe les paragraphes
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        if current and len(current) + len(paragraph) > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = ""

        current = f"{current}\n\n{paragraph}" if current else paragraph

    if current:
        chunks.append(current)

    return [
        {"title": f"Part {i + 1}", "text": chunk}
        for i, chunk in enumerate(chunks)
    ]


def load_chunks(csv_path):
    """Read the dataset's description and split it into chunks."""

    path = description_path(csv_path)

    if path is None:
        return []

    return split_into_chunks(path.read_text(encoding="utf-8"))


# --------------------------------------------------
# 2. Transformer en vecteurs
# --------------------------------------------------

def embed(texts, prefix):
    """Return one normalized vector per text."""

    response = ollama.embed(
        model=EMBED_MODEL,
        input=[prefix + t for t in texts],
    )

    vectors = np.array(response["embeddings"], dtype=np.float32)

    # Normaliser une fois pour toutes : la similarité cosinus
    # devient alors un simple produit scalaire
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / norms


# --------------------------------------------------
# 3. Construire et sauvegarder l'index d'un dataset
# --------------------------------------------------

def build_index(csv_path):
    """Embed the dataset's description and save it to <name>.rag/."""

    if description_path(csv_path) is None:
        raise FileNotFoundError(
            f"No description found next to {csv_path} "
            f"(expected {' or '.join(DESCRIPTION_EXTENSIONS)})"
        )

    chunks = load_chunks(csv_path)

    if not chunks:
        raise ValueError(f"The description of {csv_path} is empty")

    vectors = embed([c["text"] for c in chunks], DOC_PREFIX)

    folder = index_dir(csv_path)
    folder.mkdir(exist_ok=True)

    np.save(folder / "vectors.npy", vectors)
    (folder / "chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
        # Oublier l'ancienne version de cet index si elle était en cache
    _cache.pop(str(folder.resolve()), None)
    return len(chunks)


# --------------------------------------------------
# 4. Charger un index (une fois par dataset et par processus)
# --------------------------------------------------

_cache = {}


def load_index(csv_path):
    """Load a dataset's index, and refuse it if the description changed since."""

    folder = index_dir(csv_path)
    key = str(folder.resolve())

    if key in _cache:
        return _cache[key]

    vectors_file = folder / "vectors.npy"
    chunks_file = folder / "chunks.json"

    if not vectors_file.exists() or not chunks_file.exists():
        raise FileNotFoundError(
            f"No index for {csv_path}. Run: python rag.py build {csv_path}"
        )

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    vectors = np.load(vectors_file)

    # Index périmé : les morceaux sauvegardés ne correspondent plus au document
    if chunks != load_chunks(csv_path):
        raise RuntimeError(
            f"The index of {csv_path} is out of date with its description. "
            f"Run: python rag.py build {csv_path}"
        )

    _cache[key] = (chunks, vectors)

    return _cache[key]


# --------------------------------------------------
# 5. Chercher
# --------------------------------------------------

def search(csv_path, query, k=3):
    """Return the k chunks of the dataset's description closest to the query."""

    chunks, vectors = load_index(csv_path)

    query_vector = embed([query], QUERY_PREFIX)[0]

    # Vecteurs normalisés : produit scalaire = similarité cosinus
    scores = vectors @ query_vector

    best = np.argsort(scores)[::-1][:k]

    return [
        {
            "title": chunks[i]["title"],
            "text": chunks[i]["text"],
            "score": round(float(scores[i]), 3),
        }
        for i in best
    ]


# --------------------------------------------------
# Ligne de commande
# --------------------------------------------------

if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) == 2 and args[0] == "build":
        count = build_index(args[1])
        print(f"Index built: {count} chunks -> {index_dir(args[1])}")

    elif len(args) >= 3 and args[0] == "search":
        query = " ".join(args[2:])
        print("Query:", query)
        for hit in search(args[1], query):
            print(f"  {hit['score']:.3f}  {hit['title']}")

    else:
        print(__doc__)