"""C2·C3 — 값 기반 필터, 인자 받는 변환기, 필드 결합.

지금까지 코드 수정 없이는 불가능하던 요건들이다. 새 요건이 올 때마다
field_transforms.py 에 함수를 추가하던 통로를 설정으로 옮긴 것이 이 세 기능의 목적이다.
"""
import json
import textwrap

import pytest

from genon.preprocessor.facade.enrichment.json_records import JsonRecordsMapper
from genon.preprocessor.facade.enrichment.tabular_custom_fields import (
    TabularCustomFieldsMapper,
)

pytestmark = pytest.mark.unit


def _rows(tmp_path, body, payload, kind="tabular"):
    path = tmp_path / "custom_field_x.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    if kind == "tabular":
        mapper = TabularCustomFieldsMapper(
            config_file=path.name, resource_path=str(tmp_path),
            doc_type="t", extractor="tabular_mapping",
        )
        data = {"data": [{"sheet_name": "S", "data_rows": payload}]}
    else:
        mapper = JsonRecordsMapper(
            config_file=path.name, resource_path=str(tmp_path),
            doc_type="t", extractor="json_mapping",
        )
        data = payload
    return mapper.build_fields(data, "t")


_FILTER_CFG = """
    column_map:
      TITLE:  [제목]
      DEL_YN: [삭제여부]
      STATUS: [상태]
    filter:
      - {field: DEL_YN, not_in: [Y]}
      - {field: STATUS, in: [ACTIVE, PENDING]}
    text_fields: [TITLE]
"""


def test_filter_selects_records_by_value(tmp_path):
    """`required` 는 빈 값만 걸렀다 — "삭제여부 Y 인 행은 빼라"를 표현할 수 없었다."""
    rows = _rows(tmp_path, _FILTER_CFG, [
        {"제목": "살아있음", "삭제여부": "N", "상태": "ACTIVE"},
        {"제목": "삭제됨", "삭제여부": "Y", "상태": "ACTIVE"},
        {"제목": "상태밖", "삭제여부": "N", "상태": "CLOSED"},
        {"제목": "대기", "삭제여부": "n", "상태": "pending"},   # 대소문자 무시
    ])
    assert [r["TITLE"] for r in rows] == ["살아있음", "대기"]
    assert rows[0]["DEL_YN"] == "N"  # 옛 value_map 우회와 달리 원값이 파괴되지 않는다


def test_filter_is_not_fail_open(tmp_path):
    """열거하지 않은 값이 통과하던 옛 우회(value_map 빈 표준값)의 반대다."""
    assert _rows(tmp_path, _FILTER_CFG,
                 [{"제목": "신규", "삭제여부": "N", "상태": "NEW_UNKNOWN"}]) == []


def test_transform_takes_arguments_and_chains(tmp_path):
    """`"18,000원"` → 18000 처럼 두 단계가 필요한 요건이 흔하다."""
    rows = _rows(tmp_path, """
        column_map:
          TITLE:   [제목]
          FEE_AMT: [연회비]
          CODE:    [설명]
          SHORT:   [본문]
        transforms:
          FEE_AMT:
            - {name: regex_sub, pattern: "[^0-9]", repl: ""}
            - {name: to_int}
          CODE:  {name: regex_extract, pattern: "([A-Z]+-[A-Z]+-[0-9]+)"}
          SHORT: {name: truncate, length: 6, suffix: "…"}
        text_fields: [TITLE]
    """, [{"제목": "카드", "연회비": "18,000원",
           "설명": "문서 CS-HPP-0231 참고", "본문": "가나다라마바사아자차"}])
    assert rows[0]["FEE_AMT"] == 18000
    assert rows[0]["CODE"] == "CS-HPP-0231"
    assert rows[0]["SHORT"] == "가나다라마…"


def test_plain_transform_name_still_works(tmp_path):
    """기존 표기를 깨지 않는다 — 출고 설정 7개가 이 형태다."""
    rows = _rows(tmp_path, """
        column_map: {DT: [일자], T: [제목]}
        transforms: {DT: date_int_flex}
        text_fields: [T]
    """, [{"일자": "26.07.01", "제목": "x"}])
    assert rows[0]["DT"] == 20260701


def test_derive_combines_fields(tmp_path):
    """두 필드를 metadata 필드로 합치는 방법이 없었다(text_fields 는 청크 본문이다)."""
    rows = _rows(tmp_path, """
        column_map:
          BRAND:      [브랜드]
          PRODUCT_NM: [상품명]
          MISSING:    [없음]
        derive:
          DISPLAY_NM: "{{BRAND}} {{PRODUCT_NM}}"
          WITH_HOLE:  "{{BRAND}} {{MISSING}}"
        text_fields: [DISPLAY_NM]
    """, [{"브랜드": "삼성카드", "상품명": "taptap O"}])
    assert rows[0]["DISPLAY_NM"] == "삼성카드 taptap O"
    assert rows[0]["WITH_HOLE"] == "삼성카드"   # 빈 자리는 지우고 양끝을 다듬는다


def test_derive_field_is_usable_downstream(tmp_path):
    """derive 산출이 "생성 가능 필드"로 등록돼야 한다.

    아니면 text_fields 에 쓸 때 오탐 경고가 나고, chunk_prefix_fields/filter 에 쓰면
    기동이 실패한다.
    """
    rows = _rows(tmp_path, """
        column_map: {A: [에이], B: [비]}
        derive: {D: "{{A}}-{{B}}"}
        split: true
        chunk_prefix_fields: [D]
        filter:
          - {field: D, not_in: [x-y]}
        text_fields: [D]
    """, [{"에이": "a", "비": "b"}, {"에이": "x", "비": "y"}])
    assert [r["D"] for r in rows] == ["a-b"]


def test_pack_bundles_fields_into_one_json_column(tmp_path):
    """`pack` — 값 여럿을 적재 컬럼 하나에 JSON 으로 담는다.

    `derive` 로는 만들 수 없다. 문자열 치환이라 값에 따옴표가 섞이면 깨진 JSON 이 조용히
    만들어진다 — 그래서 직렬화까지 이 기능이 맡는다.
    """
    rows = _rows(tmp_path, """
        column_map:
          BENEFIT: [혜택]
          LIMIT:   [한도]
          BRAND:   [브랜드]
          MISSING: [없음]
        derive:
          DISPLAY_NM: "{{BRAND}} 카드"
        pack:
          DETAIL_JSON: [BENEFIT, LIMIT, MISSING, DISPLAY_NM]
        text_fields: [BENEFIT]
    """, [{"혜택": '연 18,000원 "무이자"', "한도": "500만원", "브랜드": "삼성"}])

    assert json.loads(rows[0]["DETAIL_JSON"]) == {
        "BENEFIT": '연 18,000원 "무이자"',
        "LIMIT": "500만원",
        "MISSING": None,        # 값이 없어도 키는 남는다(적재쪽 키 집합이 흔들리지 않는다)
        "DISPLAY_NM": "삼성 카드",   # derive 산출도 담을 수 있다(pack 이 파이프라인 마지막)
    }
    assert rows[0]["BENEFIT"] == '연 18,000원 "무이자"'  # 묶은 원천은 그대로 남는다


def test_pack_result_is_always_a_string(tmp_path):
    """dict 로 두면 경로마다 모양이 갈린다 — 행 경로는 metadata 를 그대로 청크에 싣는다."""
    rows = _rows(tmp_path, """
        column_map: {A: [에이]}
        pack: {J: [A]}
        text_fields: [A]
    """, [{"에이": "a"}])

    assert isinstance(rows[0]["J"], str)


def test_pack_json_path_gets_the_same_feature(tmp_path):
    rows = _rows(tmp_path, """
        key_map: {TITLE: [title], FEE_AMT: [fee]}
        transforms: {FEE_AMT: {name: to_int}}
        pack: {DETAIL_JSON: [TITLE, FEE_AMT]}
        text_fields: [TITLE]
    """, [{"title": "연회비 안내", "fee": "18,000원"}], kind="json")

    # 변환 뒤의 값을 담는다(정수는 JSON 숫자로 나간다)
    assert json.loads(rows[0]["DETAIL_JSON"]) == {"TITLE": "연회비 안내", "FEE_AMT": 18000}


def test_json_path_gets_the_same_features(tmp_path):
    rows = _rows(tmp_path, """
        key_map: {TITLE: [title], FEE_AMT: [fee], DEL_YN: [delYn]}
        transforms: {FEE_AMT: {name: to_int}}
        derive: {LABEL: "{{TITLE}} ({{FEE_AMT}})"}
        filter:
          - {field: DEL_YN, not_in: [Y]}
        text_fields: [LABEL]
    """, [{"title": "연회비 안내", "fee": "18,000원", "delYn": "N"},
          {"title": "삭제", "fee": "0", "delYn": "Y"}], kind="json")
    assert len(rows) == 1
    assert rows[0]["LABEL"] == "연회비 안내 (18000)"


@pytest.mark.parametrize(
    "body, expect",
    [
        ("transforms:\n  T: {name: 없는변환기}\n", "등록되지 않은"),
        ("transforms:\n  T: {name: regex_sub}\n", "인자가 필요"),
        ('transforms:\n  T: {name: regex_sub, pattern: "(["}\n', "정규식"),
        ("transforms:\n  T: {name: date_int, pattern: x}\n", "인자를 받지 않"),
        ('derive:\n  D: "{{NOPE}}"\n', "만드는 설정이 없"),
        ("filter:\n  - {field: NOPE, in: [Y]}\n", "만드는 설정이 없"),
        ("filter:\n  - {field: T}\n", "정확히 하나"),
        ("filter:\n  - {field: T, in: []}\n", "비어 있지 않은"),
        ("pack:\n  J: [NOPE]\n", "만드는 설정이 없"),
        ("pack:\n  J: T\n", "목록이어야"),
        ("pack:\n  J: []\n", "묶을 필드가 없"),
        # pack 산출을 다시 묶으면 JSON 안에 JSON 문자열이 중첩되고 적용 순서가 yaml 키
        # 순서에 조용히 의존한다.
        ("pack:\n  J: [T]\n  K: [J]\n", "만드는 설정이 없"),
        # 본문 관련 키는 pack 필드를 받지 않는다(문서형은 조용히 버리고 행 경로는 싣는다).
        ("pack:\n  J: [T]\nchunk_prefix_fields: [J]\nsplit: true\n", "쓸 수 없습니다"),
    ],
)
def test_misconfiguration_is_caught_at_startup(tmp_path, body, expect):
    """요청 때 터지면 어느 설정이 문제인지 로그만 보고는 알 수 없다."""
    with pytest.raises(ValueError, match=expect):
        _rows(tmp_path, "column_map: {T: [제목]}\ntext_fields: [T]\n" + body,
              [{"제목": "x"}])
