"""
Tests for dataset_store.py.  Run: python test_dataset_store.py

Every test gets its own temporary folder, never your real sessions/ folder.
The description tests need Ollama running with nomic-embed-text.
Exit code 0 if everything passes, 1 otherwise.
"""

import sys
import tempfile
from pathlib import Path

import dataset_store
import rag


CSV_TEXT = "Date,Open,Close\n2024-01-02,10,11\n2024-01-03,11,12\n"

DESCRIPTION = """# Sales

## Open
Open is the opening price: the price of the first trade of the day.

## Close
Close is the closing price: the price of the last trade of the day.
"""


def write(folder, name, text):
    path = Path(folder) / name
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------
# 1. Identité et idempotence (dans une session)
# --------------------------------------------------

def test_same_content_same_id(tmp, root):
    first = dataset_store.add_dataset(root, write(tmp, "uber_stock_data.csv", CSV_TEXT))
    second = dataset_store.add_dataset(root, write(tmp, "uber.csv", CSV_TEXT))

    assert first == second, (first, second)
    assert len(first) == dataset_store.ID_LENGTH
    assert len(list(root.iterdir())) == 1

    # Le premier nom est conservé
    assert dataset_store.load_meta(root, first)["original_name"] == "uber_stock_data.csv"


def test_modified_content_new_id(tmp, root):
    first = dataset_store.add_dataset(root, write(tmp, "a.csv", CSV_TEXT))
    second = dataset_store.add_dataset(root, write(tmp, "b.csv", CSV_TEXT + "2024-01-04,12,13\n"))

    assert first != second


def test_csv_is_never_overwritten(tmp, root):
    dataset_id = dataset_store.add_dataset(root, write(tmp, "a.csv", CSV_TEXT))
    stored = dataset_store.dataset_csv(root, dataset_id)
    before = stored.stat().st_mtime_ns

    dataset_store.add_dataset(root, write(tmp, "again.csv", CSV_TEXT))

    assert stored.stat().st_mtime_ns == before
    assert stored.read_text(encoding="utf-8") == CSV_TEXT


def test_csv_path_is_absolute_and_unknown_id_fails(tmp, root):
    dataset_id = dataset_store.add_dataset(root, write(tmp, "a.csv", CSV_TEXT))

    assert dataset_store.dataset_csv(root, dataset_id).is_absolute()

    try:
        dataset_store.dataset_csv(root, "0000000000000000")
    except FileNotFoundError:
        return

    raise AssertionError("an unknown dataset_id should raise FileNotFoundError")


# --------------------------------------------------
# 2. Isolation entre sessions
# --------------------------------------------------

def test_sessions_are_isolated(tmp, root):
    alice = Path(tmp) / "alice"
    bob = Path(tmp) / "bob"
    csv = write(tmp, "a.csv", CSV_TEXT)

    alice_id = dataset_store.add_dataset(alice, csv)
    bob_id = dataset_store.add_dataset(bob, csv)

    # Même contenu, même identifiant... mais deux dossiers distincts
    assert alice_id == bob_id
    assert dataset_store.dataset_csv(alice, alice_id) != dataset_store.dataset_csv(bob, bob_id)

    # Une description écrite par Alice n'apparaît pas chez Bob
    (dataset_store.dataset_dir(alice, alice_id) / "data.md").write_text("Alice's notes")
    assert dataset_store.has_description(alice, alice_id)
    assert not dataset_store.has_description(bob, bob_id)


def test_new_session_ids_are_unique(tmp, root):
    ids = {dataset_store.new_session_id() for _ in range(100)}

    assert len(ids) == 100


# --------------------------------------------------
# 3. Descriptions refusées
# --------------------------------------------------

def test_unsupported_format_is_refused(tmp, root):
    dataset_id = dataset_store.add_dataset(root, write(tmp, "a.csv", CSV_TEXT))

    try:
        dataset_store.set_description_file(root, dataset_id, write(tmp, "doc.pdf", "fake"))
    except ValueError as error:
        assert ".pdf" in str(error)
        return

    raise AssertionError("a .pdf description should be refused for now")


def test_empty_description_is_refused(tmp, root):
    dataset_id = dataset_store.add_dataset(root, write(tmp, "a.csv", CSV_TEXT))

    try:
        dataset_store.set_description_text(root, dataset_id, "   \n  ")
    except ValueError:
        assert not dataset_store.has_description(root, dataset_id)
        return

    raise AssertionError("an empty description should be refused")


# --------------------------------------------------
# 4. Descriptions acceptées (Ollama requis)
# --------------------------------------------------

def test_pasted_description_is_indexed(tmp, root):
    dataset_id = dataset_store.add_dataset(root, write(tmp, "a.csv", CSV_TEXT))

    count = dataset_store.set_description_text(root, dataset_id, DESCRIPTION)

    assert count == 2, count
    assert dataset_store.has_description(root, dataset_id)

    hits = rag.search(dataset_store.dataset_csv(root, dataset_id), "What is the opening price?", k=1)
    assert hits[0]["title"] == "Open", hits


def test_uploaded_file_and_replacement(tmp, root):
    dataset_id = dataset_store.add_dataset(root, write(tmp, "a.csv", CSV_TEXT))
    csv_path = dataset_store.dataset_csv(root, dataset_id)

    assert dataset_store.set_description_file(root, dataset_id, write(tmp, "doc.md", DESCRIPTION)) == 2

    # Remplacer la description reconstruit l'index : pas d'erreur "out of date"
    new_description = DESCRIPTION + "\n## Volume\nVolume is the number of shares traded.\n"
    assert dataset_store.set_description_file(root, dataset_id, write(tmp, "doc.txt", new_description)) == 3

    titles = [hit["title"] for hit in rag.search(csv_path, "How many shares were traded?", k=1)]
    assert titles == ["Volume"], titles


# --------------------------------------------------
# Mini lanceur de tests
# --------------------------------------------------

def main():

    tests = [
        ("same content, same id", test_same_content_same_id),
        ("modified content, new id", test_modified_content_new_id),
        ("csv is never overwritten", test_csv_is_never_overwritten),
        ("absolute csv path, unknown id fails", test_csv_path_is_absolute_and_unknown_id_fails),
        ("sessions are isolated", test_sessions_are_isolated),
        ("new session ids are unique", test_new_session_ids_are_unique),
        ("unsupported format is refused", test_unsupported_format_is_refused),
        ("empty description is refused", test_empty_description_is_refused),
        ("pasted description is indexed", test_pasted_description_is_indexed),
        ("uploaded file and replacement", test_uploaded_file_and_replacement),
    ]

    failures = 0

    for name, test in tests:

        # Chaque test a son propre dossier temporaire, vide au départ
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "session"
            try:
                test(tmp, root)
                print(f"PASS  {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL  {name}\n      {error}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()