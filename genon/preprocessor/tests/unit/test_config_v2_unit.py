"""v2 스키마 — 정규화 단위 테스트.

v2 는 새 파이프라인이 아니라 **내부 형태로 번역하는 앞단**이다. 그래서 여기서 고정할
것은 "번역이 정확한가" 하나이고, 동작 동일성은 같은 매퍼를 타는 구조가 보장한다.
"""
import textwrap

import pytest
import yaml

from genon.preprocessor.facade.enrichment import config_v2 as cv2
from genon.preprocessor.facade.enrichment.tabular_custom_fields import (
    TabularCustomFieldsMapper,
)

pytestmark = pytest.mark.unit


def test_field_spec_must_be_a_dict(tmp_path):
    """`Q:` 처럼 값을 빠뜨리면 null 로 파싱된다 — 단축 표기를 받지 않는 이유다."""
    cfg = tmp_path / "custom_field_bad.yaml"
    cfg.write_text("schema: v2\nsource: {kind: rows}\nfields:\n  Q:\n", encoding="utf-8")
    with pytest.raises(ValueError, match="object 여야"):
        TabularCustomFieldsMapper(
            config_file=cfg.name, resource_path=str(tmp_path),
            doc_type="x", extractor="tabular_mapping",
        )


def test_missing_schema_line_is_refused(tmp_path):
    """폐기된 v1 표기를 다른 해석 모드로 조용히 받지 않는다."""
    cfg = tmp_path / "custom_field_v1.yaml"
    cfg.write_text("column_map:\n  Q: [질문]\ntext_fields: [Q]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema: v2"):
        TabularCustomFieldsMapper(
            config_file=cfg.name, resource_path=str(tmp_path),
            doc_type="x", extractor="tabular_mapping",
        )


def test_empty_config_is_not_refused():
    """`config_file` 을 쓰지 않는 등록 블록은 번역할 것이 없다 — 통과해야 한다."""
    assert cv2.load({}, label="t") == ({}, None)


@pytest.mark.parametrize(
    "body, expect",
    [
        ("source:\n  kind: bogus\n", "source.kind"),
        ("source:\n  kind: rows\n  records_at: x\n", "records_at"),
        ("source:\n  kind: rows\nfields:\n  Q: {alias: [질문], typo: 1}\n", "typo"),
        ("source:\n  kind: sections\nfilter:\n  - {field: X, in: [Y]}\n", "filter"),
        # document 도 alias/values/transform/template 을 쓴다(front matter 가 두 번째 원천).
        # kind 제약이 남은 필드 스펙은 records 전용인 collect 뿐이다.
        ("source:\n  kind: document\nfields:\n  Q: {collect: [a]}\n", "collect"),
    ],
)
def test_v2_rejects_malformed_config(body, expect):
    """v2 는 모르는 키·잘못된 자리를 조용히 무시하지 않는다."""
    cfg = yaml.safe_load(textwrap.dedent("schema: v2\n" + body))
    with pytest.raises(cv2.ConfigV2Error, match=expect):
        cv2.normalize(cfg)


def test_preprocess_blocks_reach_the_internal_form():
    """markdown/html 전처리는 enricher 가 아니라 parser 가 소비한다 — 최상위로 풀린다."""
    internal, extractor = cv2.normalize({
        "schema": "v2",
        "source": {"kind": "document",
                   "pre": {"markdown": {"text_fence": True},
                           "html": {"marker_headings": True}}},
        "llm": [{"endpoint": {"url": "u", "model": "m"}, "out": ["A"]}],
    })
    assert extractor == "llm"
    assert internal["markdown"] == {"text_fence": True}
    assert internal["html"] == {"marker_headings": True}


# ── 표기 ↔ 내부 형태 드리프트 가드 ─────────────────────────────────────────

def test_v2_covers_every_v1_key():
    """내부 키를 늘리고 v2 표기를 잊으면 여기서 깨진다.

    두 층이 갈리는 가장 흔한 경로다 — 매퍼가 읽는 키를 추가하고 config_v2 의 매핑 표를
    갱신하지 않으면, 그 키는 어떤 설정으로도 만들 수 없는데 아무도 모른다.
    """
    from genon.preprocessor.facade.enrichment import config_schema as cs

    v1_keys = set().union(*cs.EXTRACTOR_KEYS.values())
    missing = sorted(v1_keys - cv2.COVERED_V1_KEYS)
    assert not missing, (
        f"v2 가 표현하지 못하는 내부 키: {missing}. config_v2 의 매핑 표에 추가하세요."
    )


def test_covered_set_has_no_phantom_keys():
    """반대 방향 — 없어진 내부 키가 covered 에 남으면 위 검사가 헛돈다."""
    from genon.preprocessor.facade.enrichment import config_schema as cs

    v1_keys = set().union(*cs.EXTRACTOR_KEYS.values()) | set(cs.WIRING_KEYS)
    phantom = sorted(cv2.COVERED_V1_KEYS - v1_keys)
    assert not phantom, f"매퍼가 읽지 않는 키가 covered 에 남아 있습니다: {phantom}"


def test_v2_config_still_gets_extractor_level_validation(tmp_path):
    """v2 는 내부 형태로 번역된 뒤 **같은 검증기**를 탄다 — 검증이 두 벌이 되지 않는다.

    json_semantic 은 chunk_prefix_fields 를 읽지 않으므로, v2 의 body.repeat 로 그 키를
    만들면 번역 결과가 extractor 지원키 검사에서 걸려야 한다.
    """
    from genon.preprocessor.facade.enrichment.json_semantic import SemanticJsonMapper

    cfg = tmp_path / "custom_field_s.yaml"
    cfg.write_text(
        "schema: v2\n"
        "source:\n  kind: sections\n  sections: {ksp: {name: 혜택, include: true}}\n"
        "fields:\n  PRODUCT_NM: {alias: [prodNm]}\n"
        "body:\n  repeat: [PRODUCT_NM]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="chunk_prefix_fields"):
        SemanticJsonMapper(
            config_file=cfg.name, resource_path=str(tmp_path),
            doc_type="t", extractor="json_semantic",
        )


def test_mapping_tables_are_single_source():
    """표기 키와 내부 키의 대응표가 한 벌인지 — 한쪽을 지우면 함께 멈춰야 한다."""
    assert set(cv2._SPEC_TO_BLOCK) <= cv2.FIELD_SPEC_KEYS
    assert set(cv2._BODY_TO_V1) == cv2.BODY_KEYS
    assert set(cv2._SOURCE_TO_V1) | {"kind", "table_at", "pre"} == cv2.SOURCE_KEYS


# ── 배포 전 점검 ────────────────────────────────────────────────────────────

def _load_script(name: str):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "config_precheck" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    import sys as _sys

    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_precheck_understands_v2_configs(tmp_path):
    """배포 전 점검이 v2 설정을 내부 키로 검사해 전건 실패로 보면 안 된다."""
    precheck = _load_script("precheck_custom_fields.py")
    (tmp_path / "custom_field_x.yaml").write_text(
        "schema: v2\nsource: {kind: rows}\nfields:\n  Q: {alias: [질문]}\n"
        "body:\n  fields: [Q]\n",
        encoding="utf-8",
    )
    block = {"doc_type": "t", "extractor": "tabular_mapping",
             "config_file": "custom_field_x.yaml"}
    assert precheck.check_block("cfg.yaml", block, tmp_path, set()) == []


def test_precheck_refuses_v1_notation(tmp_path):
    """기동에서 막히는 설정은 배포 전 점검에서도 막혀야 한다."""
    precheck = _load_script("precheck_custom_fields.py")
    (tmp_path / "custom_field_x.yaml").write_text(
        "column_map:\n  Q: [질문]\ntext_fields: [Q]\n", encoding="utf-8"
    )
    block = {"doc_type": "t", "extractor": "tabular_mapping",
             "config_file": "custom_field_x.yaml"}
    problems = precheck.check_block("cfg.yaml", block, tmp_path, set())
    assert any("schema: v2" in p and p.startswith("[기동실패]") for p in problems), problems
