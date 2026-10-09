"""Small offline tests for retrieval helpers; does not call OpenAI or download models."""
from langchain_core.documents import Document
from rag_engine import _rrf, _tokenize, _unique_docs

def doc(text, source="a.pdf", page=1, start=0):
    return Document(page_content=text, metadata={"source": source, "page": page, "start_index": start})

def test_tokenize_handles_identifiers():
    assert "sec-10k" in _tokenize("SEC-10K filing")

def test_rrf_promotes_documents_present_in_multiple_rankings():
    a, b, c = doc("alpha", start=0), doc("beta", start=1), doc("gamma", start=2)
    ranked = _rrf([[a, b], [a, c]])
    assert ranked[0].page_content == "alpha"

def test_unique_docs_removes_duplicates():
    a = doc("same", start=0)
    assert len(_unique_docs([a, doc("same", start=0)])) == 1
