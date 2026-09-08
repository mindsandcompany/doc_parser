"""표 설명 배치가 응답 상한(max_tokens)으로도 끊기고 동시에 나가는지 고정한다.

이전에는 배치를 입력 예산(max_context_tokens)만으로 묶어, 프롬프트에는 들어가지만 응답이
max_tokens 에서 잘리는 배치가 만들어졌다. 잘린 JSON 은 파싱에 실패해 그 호출의 표 전부가
설명 없이 지나갔다(실측: 표 1,784개 문서). 외부 LLM 은 부르지 않고 _call_llm 을 대체한다.
"""

import asyncio
import json
import time

import pytest

from genon.preprocessor.facade.enrichment.custom_fields_enricher import CustomFieldsEnricher
from genon.preprocessor.facade.enrichment.table_text_context import (
    TableTextDescriptionOptions,
    TableTextTarget,
)

TABLE_PROMPT = "표별 RAG 설명을 _table_descriptions JSON 배열로 반환하라."
RAG = {
    "retrieval_context_max_chars": 250,
    "key_fact_limit": 2,
    "key_fact_max_chars": 120,
    "search_terms_limit": 3,
}


def _enricher(max_tokens=16000, concurrency=8):
    return CustomFieldsEnricher(
        url="http://llm.invalid/v1/chat/completions",
        model="test-model",
        output_fields=[],
        max_tokens=max_tokens,
        table_text_description={
            "enabled": True,
            "prompt_template": TABLE_PROMPT,
            "concurrency": concurrency,
            "rag": RAG,
        },
    )


def _targets(count, table_chars=400):
    return [
        TableTextTarget(
            table_id=f"table_{i + 1:04d}", table_item=None, page_no=1,
            section_header="H", caption="", before_context="앞" * 50,
            table_text="셀" * table_chars, after_context="뒤" * 50,
            input_format="markdown",
        )
        for i in range(count)
    ]


def _echo_response(suffix):
    """프롬프트에 실린 table_id 를 그대로 되돌려주는 가짜 응답."""
    ids = [
        line.split('"')[1]
        for line in suffix.splitlines()
        if line.startswith('<table_target id=')
    ]
    return json.dumps({"_table_descriptions": [
        {"table_id": tid, "retrieval_context": "설명", "key_facts": [], "search_terms": []}
        for tid in ids
    ]}, ensure_ascii=False)


@pytest.mark.unit
def test_output_budget_sets_the_batch_ceiling():
    """1회 최대 표 개수는 max_tokens 에 비례한다."""
    per_table = TableTextDescriptionOptions.from_config({"rag": RAG}).estimated_output_chars_per_table
    assert per_table == 250 + 2 * 120 + 3 * 20 + 120

    ceilings = [_enricher(max_tokens=mt)._max_tables_per_call() for mt in (4000, 16000, 64000)]
    assert ceilings == sorted(ceilings) and ceilings[0] >= 1
    assert ceilings[1] > ceilings[0]
    # max_tokens 를 못 읽으면 상한 없음(0) — 예전 동작으로 되돌아간다.
    assert _enricher(max_tokens=0)._max_tables_per_call() == 0


@pytest.mark.unit
@pytest.mark.parametrize("max_tokens,count", [(4000, 40), (16000, 40), (16000, 200)])
def test_batches_never_exceed_the_output_ceiling_and_lose_no_table(max_tokens, count):
    enricher = _enricher(max_tokens=max_tokens)
    ceiling = enricher._max_tables_per_call()
    batches = enricher._plan_batches(None, _targets(count))
    assert sum(len(batch) for batch in batches) == count
    assert max(len(batch) for batch in batches) <= ceiling


@pytest.mark.unit
def test_single_call_is_given_up_when_output_would_overflow():
    """표가 상한을 넘으면 입력에 들어가더라도 단일 호출을 쓰지 않는다."""
    enricher = _enricher()
    ceiling = enricher._max_tables_per_call()
    assert enricher._fit_targets("", None, _targets(ceiling))[1] is True
    assert enricher._fit_targets("", None, _targets(ceiling + 1))[1] is False


@pytest.mark.unit
def test_batches_run_concurrently_and_describe_every_table():
    enricher = _enricher()
    targets = _targets(200)
    batch_count = len(enricher._plan_batches(None, targets))
    assert batch_count > 1, "이 표 개수는 여러 배치로 나뉘어야 의미가 있다"

    async def _slow(raw, document=None, user_suffix=""):
        await asyncio.sleep(0.05)
        return _echo_response(user_suffix)

    enricher._call_llm = _slow
    started = time.perf_counter()
    described = asyncio.run(enricher.describe_table_targets(targets, document=None))
    elapsed = time.perf_counter() - started

    assert len(described) == 200
    assert elapsed < batch_count * 0.05 * 0.6, "배치가 순차로 돌았다"


@pytest.mark.unit
def test_one_failed_batch_keeps_the_others_and_still_raises():
    enricher = _enricher()
    calls = {"n": 0}

    async def _flaky(raw, document=None, user_suffix=""):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("서버 오류")
        return _echo_response(user_suffix)

    enricher._call_llm = _flaky
    with pytest.raises(RuntimeError):
        asyncio.run(enricher.describe_table_targets(_targets(200), document=None))
    assert calls["n"] > 2, "실패 배치에서 멈추지 않고 나머지도 호출해야 한다"
