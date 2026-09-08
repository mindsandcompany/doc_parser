"""attachment PDF의 PyMuPDF 빈 결과 → DotsOCR 폴백 단위 테스트."""

from types import SimpleNamespace

import pytest

attachment = pytest.importorskip("facade.attachment_processor")
Document = attachment.Document
DocumentProcessor = attachment.DocumentProcessor


@pytest.mark.unit
def test_pdf_native_text_keeps_pymupdf_fast_path(monkeypatch):
    native = [
        Document(page_content="native text", metadata={"page": 0}),
        Document(page_content="", metadata={"page": 1}),
    ]

    class FakeLoader:
        def __init__(self, path, mode):
            assert path == "sample.pdf"
            assert mode == "page"

        def load(self):
            return native

    processor = object.__new__(DocumentProcessor)
    monkeypatch.setattr(attachment, "PyMuPDFLoader", FakeLoader)
    monkeypatch.setattr(
        processor,
        "_get_empty_pdf_fallback_converter",
        lambda: pytest.fail("DotsOCR fallback must not run for non-empty PDF"),
    )

    assert processor._load_pdf_page_documents("sample.pdf") is native


@pytest.mark.unit
def test_empty_pdf_falls_back_to_dotsocr_per_page(monkeypatch):
    empty_pages = [
        Document(page_content="", metadata={"page": 0}),
        Document(page_content="  ", metadata={"page": 1}),
    ]

    class FakeLoader:
        def __init__(self, path, mode):
            assert path == "converted.pdf"
            assert mode == "page"

        def load(self):
            return empty_pages

    class FakeDoclingDocument:
        def num_pages(self):
            return 2

    class FakeConverter:
        def convert(self, path, raises_on_error):
            assert path == "converted.pdf"
            assert raises_on_error is True
            return SimpleNamespace(document=FakeDoclingDocument())

    exported = {1: "첫 페이지 OCR", 2: "둘째 페이지 OCR"}
    processor = object.__new__(DocumentProcessor)
    monkeypatch.setattr(attachment, "PyMuPDFLoader", FakeLoader)
    monkeypatch.setattr(
        processor,
        "_get_empty_pdf_fallback_converter",
        lambda: FakeConverter(),
    )
    monkeypatch.setattr(
        attachment,
        "export_markdown",
        lambda document, *, page_no, **kwargs: exported[page_no],
    )

    documents = processor._load_pdf_page_documents(
        "converted.pdf",
        source_path="sample.pptx",
        compact_tables=False,
    )

    assert [doc.page_content for doc in documents] == [
        "첫 페이지 OCR",
        "둘째 페이지 OCR",
    ]
    assert [doc.metadata for doc in documents] == [
        {"source": "sample.pptx", "page": 0},
        {"source": "sample.pptx", "page": 1},
    ]


@pytest.mark.unit
def test_empty_pdf_fallback_converter_configures_dotsocr(monkeypatch):
    captured = {}

    class FakeDocumentConverter:
        def __init__(self, *, format_options):
            captured["format_options"] = format_options

    processor = object.__new__(DocumentProcessor)
    processor._empty_pdf_fallback_layout = attachment.ps.resolve_layout_settings(
        {},
        {
            "layout_model_type": "genos_layout",
            "genos_layout": {
                "endpoint": "http://dots.test/v1/chat/completions",
                "model": "dots-mocr",
                "table_fallback_enabled": False,
            },
        },
    )
    processor._empty_pdf_fallback_pdf = attachment.ps.resolve_pdf_basics(
        {"images_scale": 2, "table_structure_mode": "accurate"}
    )
    monkeypatch.setattr(attachment, "DocumentConverter", FakeDocumentConverter)

    first = processor._get_empty_pdf_fallback_converter()
    second = processor._get_empty_pdf_fallback_converter()

    assert first is second
    format_option = captured["format_options"][attachment.InputFormat.PDF]
    pdf_options = format_option.pipeline_options
    assert pdf_options.do_ocr is False
    assert pdf_options.do_table_structure is True
    assert (
        pdf_options.table_structure_options.table_structure_model_type
        == attachment.TableStructureModelType.DOTSOCR
    )
    assert pdf_options.layout_options.layout_model_type.value == "genos_layout"
    assert (
        pdf_options.layout_options.genos_layout_options.endpoint
        == "http://dots.test/v1/chat/completions"
    )
    assert pdf_options.layout_options.genos_layout_options.model == "dots-mocr"
    assert pdf_options.layout_options.genos_layout_options.table_fallback_enabled is False
    assert pdf_options.images_scale == 2
    assert pdf_options.generate_page_images is False
    assert pdf_options.generate_picture_images is False


@pytest.mark.unit
def test_load_documents_routes_pdf_to_empty_result_fallback(monkeypatch):
    expected = [Document(page_content="OCR text", metadata={"page": 0})]
    processor = object.__new__(DocumentProcessor)
    monkeypatch.setattr(processor, "get_real_file_type", lambda path: "pdf")
    monkeypatch.setattr(
        processor,
        "_load_pdf_page_documents",
        lambda path, **kwargs: expected,
    )
    monkeypatch.setattr(
        processor,
        "get_loader",
        lambda *args, **kwargs: pytest.fail("PDF must use the fallback-aware loader"),
    )

    assert processor.load_documents("sample.pdf") is expected
