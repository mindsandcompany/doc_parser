from pathlib import Path
import pytest

@pytest.mark.smoke
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "sample_files" / "sample.pdf").exists(),
    reason="sample.pdf not found",
)
def test_pdf_smoke(basic_processor):
    dp = basic_processor()
    sample = Path(__file__).resolve().parents[2] / "sample_files" / "sample.pdf"

    doc = dp.load_documents(str(sample))
    assert doc is not None
    if hasattr(doc, "num_pages"):
        assert doc.num_pages() >= 1

    chunks = dp.split_documents(doc)
    assert isinstance(chunks, list) and len(chunks) >= 1

@pytest.mark.smoke
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "sample_files" / "sample.pdf").exists(),
    reason="sample.pdf not found",
)
@pytest.mark.asyncio
async def test_vector_schema_pdf(basic_processor):
    dp = basic_processor()
    sample = Path(__file__).resolve().parents[2] / "sample_files" / "sample.pdf"

    vectors = await dp(None, str(sample))
    assert isinstance(vectors, list) and len(vectors) >= 1
    v = vectors[0]
    if hasattr(v, "model_dump"):
        v = v.model_dump()
    required = [
        "text",
        "n_char",
        "n_word",
        "n_line",
        "i_page",
        "i_chunk_on_page",
        "i_chunk_on_doc",
    ]
    for k in required:
        assert k in v, f"missing key: {k}"
    assert isinstance(v["text"], str)
    for k in [x for x in required if x != "text"]:
        assert isinstance(v[k], int), f"{k} should be int"
