"""
Tests for rag.py.  Run: python test_rag.py

The retrieval tests need Ollama running with nomic-embed-text.
Exit code 0 if everything passes, 1 otherwise.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import rag


CSV = "uber_stock_data.csv"


# --------------------------------------------------
# 1. Découpage (pas besoin d'Ollama)
# --------------------------------------------------

def test_split_on_headings():
    text = "# Title\n\n## Open\nOpen is the opening price.\n\n## Close\nClose is the closing price."
    chunks = rag.split_into_chunks(text)

    assert [c["title"] for c in chunks] == ["Open", "Close"], chunks
    assert chunks[0]["text"] == "Open\nOpen is the opening price.", chunks[0]


def test_split_free_text_groups_paragraphs():
    short = "Salary is the yearly salary of the employee."
    long = "word " * 200   # 1000 caractères, plus que MAX_CHUNK_CHARS

    chunks = rag.split_into_chunks(f"{short}\n\n{short}\n\n{long}")

    # Les deux paragraphes courts sont regroupés, le long reste seul
    assert len(chunks) == 2, [len(c["text"]) for c in chunks]
    assert chunks[0]["text"] == f"{short}\n\n{short}"


def test_uber_description_has_11_sections():
    titles = [c["title"] for c in rag.load_chunks(CSV)]

    assert len(titles) == 11, titles
    assert "Adj Close" in titles


# --------------------------------------------------
# 2. Convention de nommage
# --------------------------------------------------

def test_description_and_index_paths():
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "sales.csv"
        csv.write_text("a\n1\n")

        assert rag.description_path(csv) is None

        (Path(tmp) / "sales.txt").write_text("Sales description.")

        assert rag.description_path(csv).name == "sales.txt"
        assert rag.index_dir(csv).name == "sales.rag"


# --------------------------------------------------
# 3. Garde-fous
# --------------------------------------------------

def test_missing_index_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "sales.csv"
        csv.write_text("a\n1\n")
        (Path(tmp) / "sales.md").write_text("## Amount\nAmount is the sale value.")

        try:
            rag.load_index(csv)
        except FileNotFoundError:
            return

        raise AssertionError("load_index should refuse a dataset without an index")


def test_stale_index_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "sales.csv"
        csv.write_text("a\n1\n")
        (Path(tmp) / "sales.md").write_text("## Amount\nAmount is the sale value.")

        # Un index qui correspond à une ANCIENNE version de la description
        folder = rag.index_dir(csv)
        folder.mkdir()
        old_chunks = [{"title": "Amount", "text": "Amount\nOld text."}]
        (folder / "chunks.json").write_text(json.dumps(old_chunks))
        np.save(folder / "vectors.npy", np.zeros((1, 3), dtype=np.float32))

        try:
            rag.load_index(csv)
        except RuntimeError:
            return

        raise AssertionError("load_index should refuse an out-of-date index")


# --------------------------------------------------
# 4. Recherche (Ollama requis)
#    (question, titres attendus, dans le top N)
# --------------------------------------------------

RETRIEVAL_CASES = [
    ("What is the difference between Close and Adj Close?", {"Close", "Adj Close"}, 2),
    ("How many shares were traded?", {"Volume"}, 3),
    ("When did Uber go public?", {"IPO"}, 1),
    ("How much did the price move within a single day?", {"Daily range and volatility"}, 3),
]


def make_retrieval_test(question, expected, top):

    def test():
        titles = [hit["title"] for hit in rag.search(CSV, question, k=top)]
        assert expected <= set(titles), f"expected {sorted(expected)} in top {top}, got {titles}"

    return test


# --------------------------------------------------
# Mini lanceur de tests
# --------------------------------------------------

def main():

    tests = [
        ("split on headings", test_split_on_headings),
        ("split free text", test_split_free_text_groups_paragraphs),
        ("uber description has 11 sections", test_uber_description_has_11_sections),
        ("description and index paths", test_description_and_index_paths),
        ("missing index is refused", test_missing_index_is_refused),
        ("stale index is refused", test_stale_index_is_refused),
    ]

    for question, expected, top in RETRIEVAL_CASES:
        tests.append((f"retrieve: {question}", make_retrieval_test(question, expected, top)))

    # Index frais avant les tests de recherche
    print(f"Building index for {CSV}...")
    rag.build_index(CSV)

    failures = 0

    for name, test in tests:
        try:
            test()
            print(f"PASS  {name}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL  {name}\n      {error}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()