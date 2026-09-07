from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_parse_created_date():
    pass


@pytest.mark.unit
def test_safe_join():
    pass


# ─── _get_pdf_path ────────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("ext", [".hwp", ".txt", ".json", ".md", ".ppt", ".pptx", ".docx"])
def test_get_pdf_path_returns_pdf_for_convertible_ext(ext):
    from facade.parser_processor import _get_pdf_path
    assert _get_pdf_path(f"/path/to/file{ext}") == "/path/to/file.pdf"


@pytest.mark.unit
def test_get_pdf_path_preserves_directory_structure():
    from facade.parser_processor import _get_pdf_path
    assert _get_pdf_path("/some/deep/dir/document.docx") == "/some/deep/dir/document.pdf"


@pytest.mark.unit
def test_get_pdf_path_pdf_input_is_unchanged():
    from facade.parser_processor import _get_pdf_path
    assert _get_pdf_path("/path/to/file.pdf") == "/path/to/file.pdf"


# ─── convert_to_pdf (subprocess argument verification) ───────────────────────
# soffice 를 실제로 부르는 곳은 backend 모듈이다. facade 의 convert_to_pdf 는
# facade/common/pdf_convert.py 를 거쳐 그 backend 로 위임하므로, subprocess 는
# 호출이 일어나는 모듈(converters.hwp_to_pdf.libreoffice)에서 가로채야 한다.
# 예전에는 facade.parser_processor.subprocess 를 patch 했고, 그 mock 을 유지하려고
# parser 만 backend 를 안 쓰고 soffice 를 직접 부르는 사본을 들고 있었다(#199).
_SOFFICE_RUN = "genon.preprocessor.converters.hwp_to_pdf.libreoffice.subprocess.run"
# soffice 가용성 관문이 두 곳이다. facade 의 convert_to_pdf 가 지나는 이슈 #286 사전 체크와,
# backend chain 을 구성하는 hwp_to_pdf.config 의 _AVAILABILITY 다. 둘 다 두면 이 테스트가
# "이 기계에 LibreOffice 가 깔려 있는가" 를 함께 보게 되어, 없는 환경에서는 subprocess mock 에
# 닿지도 못하고 None 이 나온다(실측: CI 8건 실패, 로컬은 brew soffice 가 있어 통과했다).
#
# 두 관문의 뿌리는 같은 `shutil.which("soffice")` 다. config 는 import 시점에 함수 객체를
# _AVAILABILITY 에 담으므로 libreoffice_available 을 패치해도 잡히지 않는다 - OS probe 를
# 패치해야 두 곳이 함께 잡힌다. 여기서 보려는 것은 확장자별 convert-to 인자뿐이다.
_LO_WHICH = "genon.preprocessor.converters.hwp_to_pdf.availability.shutil.which"

@pytest.mark.unit
@pytest.mark.parametrize("ext,expected_arg", [
    (".pptx", "pdf:impress_pdf_Export"),
    (".ppt",  "pdf:impress_pdf_Export"),
    (".docx", "pdf:writer_pdf_Export"),
    (".doc",  "pdf:writer_pdf_Export"),
    (".xlsx", "pdf:calc_pdf_Export"),
    (".xls",  "pdf:calc_pdf_Export"),
    (".csv",  "pdf:calc_pdf_Export"),
    (".txt",  "pdf"),
])
def test_convert_to_pdf_passes_correct_convert_arg(ext, expected_arg, tmp_path):
    from facade.parser_processor import convert_to_pdf

    in_file = tmp_path / f"test{ext}"
    in_file.write_bytes(b"fake content")
    (tmp_path / "test.pdf").write_bytes(b"fake pdf")  # pre-create so exists() is True

    with patch(_LO_WHICH, return_value="/usr/bin/soffice"), patch(_SOFFICE_RUN) as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = convert_to_pdf(str(in_file))

    assert result is not None
    called_cmd = mock_run.call_args[0][0]
    assert expected_arg in called_cmd


@pytest.mark.unit
def test_convert_to_pdf_returns_none_when_soffice_fails(tmp_path):
    from facade.parser_processor import convert_to_pdf

    in_file = tmp_path / "test.docx"
    in_file.write_bytes(b"fake content")

    # 가용성을 고정하지 않으면 LibreOffice 가 없는 환경에서 "soffice 가 실패해서" 가 아니라
    # "사전 체크에 걸려서" None 이 되어, 통과하지만 아무것도 보지 않는 상태가 된다.
    with patch(_LO_WHICH, return_value="/usr/bin/soffice"), patch(_SOFFICE_RUN) as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="conversion failed")
        result = convert_to_pdf(str(in_file))

    assert result is None
