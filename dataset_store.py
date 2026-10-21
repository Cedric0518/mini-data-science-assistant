"""
Content-addressed storage for uploaded datasets.

    datasets/<dataset_id>/
        data.csv     the uploaded CSV, never modified
        data.md      its description (optional, replaceable)
        data.rag/    the description's index, built by rag.py
        meta.json    original file name and upload time

The dataset_id is the start of the SHA-256 of the CSV bytes:
the same file always gets the same id, a modified file a new one.
This module knows nothing about the user interface.
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import rag


STORE_DIR = Path(__file__).resolve().parent / "datasets"
ID_LENGTH = 16
DESCRIPTION_EXTENSIONS = (".md", ".txt")


# --------------------------------------------------
# 1. Identité : l'empreinte du contenu
# --------------------------------------------------

def file_fingerprint(path):
    """SHA-256 of the file bytes, read in blocks so big files fit in memory."""

    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def dataset_dir(dataset_id):
    return STORE_DIR / dataset_id


def dataset_csv(dataset_id):
    """Absolute path of the dataset's CSV. Raises if the dataset is unknown."""

    path = dataset_dir(dataset_id) / "data.csv"

    if not path.exists():
        raise FileNotFoundError(f"Unknown dataset: {dataset_id}")

    return path


# --------------------------------------------------
# 2. Ajouter un CSV (idempotent)
# --------------------------------------------------

def add_dataset(csv_path, original_name=None):
    """
    Store a CSV under its content fingerprint and return its dataset_id.
    Adding the same content again does nothing and keeps the first name.
    """

    dataset_id = file_fingerprint(csv_path)[:ID_LENGTH]
    folder = dataset_dir(dataset_id)
    target = folder / "data.csv"

    if target.exists():
        return dataset_id

    folder.mkdir(parents=True, exist_ok=True)

    # Copier sous un nom temporaire puis renommer : data.csv n'existe
    # jamais à moitié écrit, même si le programme s'arrête en pleine copie
    temporary = folder / "data.csv.tmp"
    shutil.copyfile(csv_path, temporary)
    temporary.replace(target)

    meta = {
        "original_name": original_name or Path(csv_path).name,
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return dataset_id


def load_meta(dataset_id):
    path = dataset_dir(dataset_id) / "meta.json"

    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------
# 3. Description : tout devient data.md, puis on indexe
# --------------------------------------------------

def set_description_text(dataset_id, text):
    """
    Save a pasted description as data.md and rebuild its index.
    Returns the number of indexed chunks.
    """

    csv_path = dataset_csv(dataset_id)

    text = (text or "").strip()

    if not text:
        raise ValueError("The description is empty.")

    (csv_path.parent / "data.md").write_text(text + "\n", encoding="utf-8")

    return rag.build_index(csv_path)


def set_description_file(dataset_id, doc_path):
    """Read an uploaded .md or .txt description and store it like pasted text."""

    suffix = Path(doc_path).suffix.lower()

    if suffix not in DESCRIPTION_EXTENSIONS:
        raise ValueError(
            f"Unsupported description format: {suffix or 'no extension'}. "
            f"Use {' or '.join(DESCRIPTION_EXTENSIONS)}, or paste the text."
        )

    try:
        text = Path(doc_path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError("The description file is not valid UTF-8 text.")

    return set_description_text(dataset_id, text)


def has_description(dataset_id):
    return (dataset_dir(dataset_id) / "data.md").exists()
