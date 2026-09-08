"""extractor: python — 값을 만드는 주체가 LLM 이 아니라 고객 함수인 경우 (#363 09 B군 ⑧).

지금까지 값을 뽑는 방법은 LLM·엑셀 열 매핑·JSON 키 매핑 3종 고정이었다. 정규식 추출이나
사내 마스터 조회는 코어를 고치지 않고는 넣을 수 없었다.

핵심 계약은 하나다 — **값을 만드는 주체만 다르고 그 뒤는 llm 과 완전히 같은 경로**다.
출력 필드 정리, defaults/constants, value_map/transforms/derive, 문서 저장까지 같은
코드가 돌아야 설정으로 하던 것과 코드로 하는 것이 어긋나지 않는다.

네트워크·LLM 을 부르지 않는다.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

pytestmark = pytest.mark.unit

cfe = pytest.importorskip("genon.preprocessor.facade.enrichment.custom_fields_enricher")
cs = pytest.importorskip("genon.preprocessor.facade.enrichment.config_schema")
cv2 = pytest.importorskip("genon.preprocessor.facade.enrichment.config_v2")
pl = pytest.importorskip("genon.preprocessor.facade.enrichment.plugin_loader")

_EXTRACTOR_PY = '''
def extract(text, document=None, output_fields=None, **kwargs):
    return {"CODE": text.strip().upper(), "SEEN_DOC_TYPE": kwargs.get("doc_type")}


async def extract_async(text, **kwargs):
    return {"CODE": "ASYNC"}


def extract_narrow(text):
    """넓은 시그니처를 못 받는 함수도 (text) 하나로 불린다."""
    return {"CODE": "NARROW"}


def not_a_dict(text, **kwargs):
    return ["nope"]
'''


def _site(tmp_path: Path, config: dict, py: str = _EXTRACTOR_PY) -> tuple[str, str]:
    """설정 yaml 과 고객 파이썬 파일을 같은 폴더에 둔다(경로 해석 기준)."""
    (tmp_path / "site_extract.py").write_text(py, encoding="utf-8")
    name = "custom_field_site.yaml"
    (tmp_path / name).write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return name, str(tmp_path)


def _enricher(tmp_path: Path, *, callable_name: str = "extract", **config_extra):
    config = {"file": "site_extract.py", "callable": callable_name,
              "output_fields": ["CODE"], **config_extra}
    config_file, resource_path = _site(tmp_path, config)
    return cfe.CustomFieldsEnricher(
        doc_type="site", extractor="python",
        config_file=config_file, resource_path=resource_path,
    )


def _run(enricher, text="abc", **kwargs):
    """enrich 를 돌리고 문서에 저장된 metadata 를 돌려준다."""
    doc = MagicMock()
    enricher._extract_raw_text = MagicMock(return_value=text)
    stored: list = []
    original = cfe.store_metadata_in_document
    cfe.store_metadata_in_document = lambda document, metadata, **kw: stored.append(metadata)
    try:
        asyncio.run(enricher.enrich(doc, doc_type="site", **kwargs))
    finally:
        cfe.store_metadata_in_document = original
    return stored[0] if stored else {}


# ---------------------------------------------------------------------------
# 등록 — 문서 단위 extractor 로 인정된다
# ---------------------------------------------------------------------------

def test_python_is_a_document_scope_extractor():
    assert "python" in cfe.DOCUMENT_CUSTOM_FIELD_EXTRACTORS
    assert "python" in cfe.SUPPORTED_CUSTOM_FIELD_EXTRACTORS


def test_builder_creates_an_enricher_for_python(tmp_path: Path):
    config_file, resource_path = _site(tmp_path, {"file": "site_extract.py"})
    enrichers = cfe.build_document_custom_fields_enrichers([
        {"extractor": "python", "config_file": config_file, "resource_path": resource_path},
    ])
    assert len(enrichers) == 1 and enrichers[0].is_configured is True


def test_missing_file_fails_at_startup(tmp_path: Path):
    """오설정은 첫 요청이 아니라 기동에서 드러나야 한다."""
    config_file, resource_path = _site(tmp_path, {"file": "없는파일.py"})
    with pytest.raises(FileNotFoundError):
        cfe.CustomFieldsEnricher(
            extractor="python", config_file=config_file, resource_path=resource_path)


def test_path_escape_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="허용 범위"):
        pl.load_callable(tmp_path, "../밖.py", "extract", label="t")


# ---------------------------------------------------------------------------
# 호출 계약
# ---------------------------------------------------------------------------

def test_callable_receives_text_and_request_params(tmp_path: Path):
    stored = _run(_enricher(tmp_path), text=" abc ")
    assert stored["CODE"] == "ABC"


def test_narrow_signature_is_supported(tmp_path: Path):
    stored = _run(_enricher(tmp_path, callable_name="extract_narrow"))
    assert stored["CODE"] == "NARROW"


def test_async_callable_is_awaited(tmp_path: Path):
    """사내 API 조회처럼 외부 호출이 필요한 추출기를 쓸 수 있어야 한다."""
    stored = _run(_enricher(tmp_path, callable_name="extract_async"))
    assert stored["CODE"] == "ASYNC"


def test_non_dict_result_is_reported_not_stored(tmp_path: Path):
    """계약 위반은 추출 실패로 흡수한다 — 문서 전체가 죽지는 않는다."""
    stored = _run(_enricher(tmp_path, callable_name="not_a_dict"))
    assert stored == {"CODE": None}


# ---------------------------------------------------------------------------
# llm 과 같은 값 파이프라인을 탄다 — 이 항목이 이번 작업의 핵심이다
# ---------------------------------------------------------------------------

def test_value_pipeline_is_shared_with_llm(tmp_path: Path):
    stored = _run(_enricher(
        tmp_path,
        constants={"SRC": "REGEX"},
        defaults={"MISSING": "기본값"},
        transforms={"CODE": ["text_norm"]},
    ))
    assert stored["SRC"] == "REGEX"        # const
    assert stored["MISSING"] == "기본값"    # default
    # 추출기는 "ABC" 를 돌려줬는데 text_norm(대소문자·공백 정규화)이 걸려 "abc" 가 된다.
    assert stored["CODE"] == "abc"


def test_output_fields_filter_applies(tmp_path: Path):
    """선언한 필드만 남긴다(추출기가 더 많이 돌려줘도)."""
    stored = _run(_enricher(tmp_path))
    assert set(stored) == {"CODE"}          # SEEN_DOC_TYPE 은 output_fields 밖


def test_table_descriptions_are_not_fused_for_python(tmp_path: Path):
    """표 설명은 LLM 이 만든다 — python 추출기는 그 호출을 하지 않는다."""
    enricher = _enricher(tmp_path)
    assert enricher.wants_table_descriptions(doc_type="site", table_text_desc=1) is False


# ---------------------------------------------------------------------------
# 설정 표기 — 기동 검증과 v2
# ---------------------------------------------------------------------------

def test_llm_only_keys_are_rejected_for_python():
    """다른 extractor 의 키를 쓰면 읽히지 않으므로 기동에서 막는다."""
    with pytest.raises(ValueError, match="쓸 수 없는 키"):
        cs.validate_known_keys({"url": "http://x"}, label="t", extractor="python")


def test_python_keys_are_declared():
    keys = cs.EXTRACTOR_KEYS["python"]
    assert {"file", "callable", "output_fields", "transforms"} <= keys
    assert "url" not in keys and "system_prompt" not in keys


def test_v2_notation_round_trips():
    v1 = {"file": "site_extract.py", "callable": "extract",
          "output_fields": ["CODE"], "constants": {"SRC": "REGEX"}}
    v2 = cv2.to_v2(v1, "python")
    assert v2["source"] == {"kind": "document"}
    assert v2["python"] == {"file": "site_extract.py", "callable": "extract", "out": ["CODE"]}
    back, _ = cv2.load(dict(v2), label="t")
    assert back == v1


def test_v2_rejects_llm_and_python_together():
    bad = {"schema": "v2", "source": {"kind": "document"},
           "python": {"file": "a.py"},
           "llm": [{"endpoint": {"url": "http://x", "model": "m"}}]}
    with pytest.raises(cv2.ConfigV2Error, match="함께 쓸 수 없습니다"):
        cv2.load(bad, label="t")


def test_v2_python_is_document_only():
    bad = {"schema": "v2", "source": {"kind": "rows"}, "python": {"file": "a.py"}}
    with pytest.raises(cv2.ConfigV2Error, match="document 전용"):
        cv2.load(bad, label="t")
