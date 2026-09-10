# 09-B. 특이 케이스 20건 — facade 코드로 어디까지 되는가

[09-requirements.md](09-requirements.md) 의 업무 요건 14건과 달리, 이 문서는 **현장에서
실제로 물어 온 처리 기법 20건**을 대상으로 한다. "이 요건을 facade 코드에서 풀 수 있는가"
를 판정하고, 안 되면 어떤 방식이면 되는지를 적는다. 판정은 전부 **실행해서** 확인했다.

판정 기준(요청자 지정)

1. 되도록 **facade 코드**(파일 2개)에서 푼다. yaml 설정으로 푸는 것은 최소화한다.
2. facade 코드만으로 불가능하면 어떤 방식이면 되는지를 제안한다.
3. 각 케이스는 매뉴얼의 개발 예시로 옮길 수 있게 정리한다.

## 검증 방법

`facade/*_processor.py` 를 상속한 스크래치 파사드를 만들어 **CLI 로 실제 파싱·청킹**했다.
훅만 검증하는 단위 테스트와 달리 산출(elements/청크 행)을 눈으로 확인했다.

```bash
PYTHONPATH=<repo>:<repo>/genon/preprocessor:<repo>/genon/preprocessor/src:\
<repo>/genon/preprocessor/.venv/lib/python3.13/site-packages \
  .venv/bin/python <스크래치 파사드>.py <원천> --doc-type <타입> -o parsed.json
```

행 조작(분리·결합·메타 제거)은 pytest 로 고정했다(아래 G 절).

## 요약

| # | 케이스 | 판정 | 자리 | 실측 |
|---|---|---|---|---|
| A1 | xlsx 컬럼에 조각난 JSON 문자열 결합 | ○ | `pre_source`(격자) | 기존 테스트로 확인 |
| A2 | xlsx 표 밖 본문 — 표만 뽑기 | ○ | `pre_source` | ✔ 실행 |
| A3 | xlsx 표 밖 본문 — 표와 함께 싣기 | △ 훅 불가 | `route_tabular` 오버라이드 | ✔ 실행 |
| B1 | 복잡한 JSON 전체 평문화 → 청크·LLM | △ 훅 불가 | `ROUTES` + 자체 라우트 | ✔ 실행 |
| B2 | JSON 일부 항목만 평문화 | △ 훅 불가 | 같은 라우트에서 항목 선택 | ✔ 실행 |
| B3 | 단순화된 JSON 을 만드는 전처리 | ○/△ | `pre_source`(설정 매칭 시) | ✔ 실행(대조군) |
| B4 | 특정 항목만 html·latex 인 경우 | ○ | `json_to_markdown(html_renderer=)` | ✔ 실행 |
| C1 | md LaTeX 수식 | ◎ 이미 됨 | 코드·설정 불필요 | ✔ 실행 |
| C2 | md 본문 내 표를 단독 표 청크로 | ◎ 이미 됨 | `chunking.table_as_chunk`(기본 on) | ✔ 실행 |
| D1 | html 섹션 헤더 지정 방식 지정 | ○ | `pre_source`(원문 문자열) | ✔ 실행 |
| E1 | 메타 특수문자 제거·정규식·포맷 변환 | ○ | `post_parse` + `tb.*` | ✔ 실행 |
| E2 | 메타 복합 후처리(문서 단위) | ○ | `post_parse` | ✔ 실행 |
| F1 | 본문에서 정규식으로 메타 추출 | ○ | `post_parse` + `set_chunk_metadata` | ✔ 실행 |
| F2 | 그 밖의 방식(LLM·사내 API·코드표) | ○ | `async post_parse` | 코드 근거 |
| F3 | **청크마다 다른** 메타 부착 | △ | `post_chunk`(훅 계약 한계) | ✔ pytest |
| G1 | 최종 청크 중 특정 청크 분리 | ○ | `post_chunk` | ✔ pytest |
| G2 | 작은 청크·특정 패턴 청크 결합 | ○ | `post_chunk` | ✔ pytest |
| G3 | 청크 메타 일괄 변환 | ○ | `post_chunk` | ✔ pytest |
| G4 | 청크 메타 일부 제거 | ○ | `post_chunk` | ✔ pytest |
| H1 | docling 포맷 후처리 | ○ | `post_parse` / `pre_chunk` | 코드 근거 |
| H2 | element 포맷 후처리 | ○ | `post_parse` / `pre_chunk` | 코드 근거 |

◎ 이미 됨 · ○ facade 코드로 됨 · △ 훅만으로는 안 되고 라우트/단계 오버라이드가 필요

**20건 중 훅 5종만으로 끝나는 것은 11건이다.** 4건은 라우트나 단계를 오버라이드해야 하고
(A3·B1·B2·F3), 2건은 이미 되고 있으며, 3건은 코드 근거로만 판정했다(F2·H1·H2).
**yaml 이 반드시 필요한 것은 0건이다** — LLM 호출만 예외로, 아래 B1 에 적었다.

---

## A. xlsx 전처리

### A1. 컬럼에 조각나 들어온 JSON 문자열 결합

예: AI 차트뷰 원천이 한 셀에 다 못 담아 `detail_1` `detail_2` … 로 쪼갠 JSON.

`pre_source` 가 `.xlsx` 를 **격자**(`{시트명: [[셀,…],…]}`)로 받는다. 병합 셀은 이미
펴진 상태이고, 돌려준 격자는 `normalize_sheets` 로 표준형이 된 뒤 그대로 파싱에 들어간다.

```python
def pre_source(self, ext, doc_type, data, work_dir=None, **kwargs):
    if ext != ".xlsx" or doc_type != "ai_chart":
        return data
    out = {}
    for name, rows in data.items():
        head, body = rows[0], rows[1:]
        parts = [i for i, c in enumerate(head) if str(c).startswith("detail_")]
        keep = [i for i in range(len(head)) if i not in parts[1:]]
        merged = [[head[i] for i in keep]]
        for r in body:
            joined = "".join(str(r[i]) for i in parts if i < len(r))
            merged.append([(joined if i == parts[0] else r[i]) for i in keep])
        out[name] = merged
    return out
```

결합한 값이 JSON 문자열이면 본문으로 쓰기 전에 평문화한다 — `tb.text(v)` 가 JSON/HTML/평문을
자동 판별하고, `tb.json_to_markdown(json.loads(v))` 은 `##` 섹션까지 만들어 청킹 경계를 준다.
**같은 변환기를 yaml 의 `transform:` 도 부르므로 설정과 코드가 어긋나지 않는다.**

### A2·A3. 표 밖 본문이 있는 시트

실측이 필요한 자리였다. 아래 시트로 확인했다(1행 안내문 / 3~5행 표 / 7행 안내문).

| 설정 | 결과 |
|---|---|
| `multi_table: false` (출고 기본) | **표 밖 본문이 표의 데이터 행으로 섞여 들어간다** — 7행이 `\| 문의: 고객센터 … \| \| \|` 로 표 행이 됐다 |
| `multi_table: true` | 표 2건만 남고 **표 밖 본문 2건이 통째로 사라진다**(헤더 판정에서 데이터 행이 없는 블록이 버려진다) |

즉 설정 어느 쪽으로도 "표 + 본문" 은 안 된다. facade 코드로 푼다. **표만 원하면 `pre_source`
한 곳이면 되고(A2), 본문도 같이 실으려면 라우트를 오버라이드해야 한다(A3)** — `pre_source` 는
격자를 돌려주는 계약이라 본문을 element 로 낼 자리가 없기 때문이다.

```python
def _is_prose(row):
    filled = [c for c in row if str(c).strip()]
    return len(filled) == 1 and len(str(filled[0])) > 15

class DocumentProcessor(Base):
    def pre_source(self, ext, doc_type, data, work_dir=None, **kwargs):   # A2: 표만
        if ext not in (".xlsx", ".xlsm", ".csv"):
            return data
        return {n: [r for r in rows if not _is_prose(r)] for n, rows in data.items()}

    async def route_tabular(self, file_path, ext, ctx, **kwargs):          # A3: 본문도 함께
        result = await super().route_tabular(file_path, ext, ctx, **kwargs)
        prose = [str([c for c in r if str(c).strip()][0])
                 for rows in tb.load_sheets(file_path).values()
                 for r in rows if _is_prose(r)]
        if prose:
            result.setdefault("elements", []).extend(tb.make_elements(prose))
        return result
```

실측 산출(출고 기본 설정 그대로) — `tabular_row` 2건 + `paragraph` 2건.

```
tabular_row | '시트명: 안내\n| 상품 | 금리 | 비고 |\n| 정기예금 | 3.10 | 기본 |'
tabular_row | '시트명: 안내\n| 상품 | 금리 | 비고 |\n| 적금 | 2.85 | 우대별도 |'
paragraph   | '2026년 3분기 상품 안내문입니다. 아래 표를 참고하세요.'
paragraph   | '문의: 고객센터 1588-0000. 표 밖 본문 두 번째 문단입니다.'
```

**주의 — `document` 와 `elements` 는 공존하지 못한다.** 청커의 `_classify_payload` 가
`document` 를 먼저 보므로, 문서형 산출에 element 를 덧붙여도 청커는 element 를 무시한다.
"표는 행으로, 본문은 문단으로" 처럼 섞고 싶으면 **한쪽 표현으로 통일**해야 한다
(위 레시피는 elements 로 통일한 것이다).

---

## B. 복잡한 JSON

### 먼저 확인한 구조적 사실

**custom_fields 설정이 매칭되지 않는 `.json` 은 `pre_source` 를 한 번도 지나지 않는다.**
`route_json` 이 설정을 못 찾으면 파일을 읽기도 전에 `None` 을 돌려 캐치올(TextLoader)로
빠지고, `.json` 은 경로형 훅 대상에서도 빠져 있기 때문이다(`_DATA_HOOK_EXTS`).

```python
out = await p.route_json(str(path), ".json", {"enrichment_context": {}})
assert out is None      # 캐치올로 폴백
assert seen == []       # pre_source 는 한 번도 안 불렸다
```

그래서 **JSON 을 facade 코드로 다루려면 `ROUTES` 에 자기 라우트를 한 줄 얹는 것이 출발점**이다
(`ROUTES` 는 facade 소유라 코드 수정 범위 안이다).

### B1·B2·B4. 전체/일부 평문화, html·latex 가 섞인 항목

```python
class DocumentProcessor(Base):
    ROUTES = (((".json",), "route_json_flat"),) + Base.ROUTES

    async def route_json_flat(self, file_path, ext, ctx, **kwargs):
        if tb.normalize_doc_type(kwargs.get("doc_type")) != "ins_api":
            return None                                   # 다른 타입은 기존 경로로
        payload = json.loads(tb.read_text_with_fallback(file_path))
        picked = [it for it in payload["items"] if it.get("type") == "product"]  # B2: 일부만
        md = tb.json_to_markdown(picked, html_renderer=structural_html_renderer())
        with tempfile.TemporaryDirectory(prefix="json_flat_") as work_dir:
            md_path = os.path.join(work_dir, os.path.basename(file_path) + ".md")
            open(md_path, "w", encoding="utf-8").write(md)
            return await self.route_docling(md_path, ".md", ctx, **kwargs)   # 문서 경로에 태운다
```

실측 산출 — 중첩 배열이 `##` 섹션과 목록으로 펼쳐지고, 항목 안의 **HTML 표는 진짜 표
(`tables: 1`)로, LaTeX 는 `$$…$$` 그대로** 살아남았다.

```
section_header '1' / list_item 'code: L2026-014' / section_header 'detail'
text '주계약' / text '사망보험금' / text '- formula: $$V_t = P \times a_{x+t}$$'
section_header 'coverages 1' / list_item 'pay Reasons.pay Reasons 1.rate: 100%'
tables: 1
```

핵심은 마지막 줄의 `route_docling` 위임이다. 직접 element 를 만들면 표·헤딩 구조와
enrichment(표 설명·custom_fields)를 전부 포기하게 되는데, md 로 한 번 내려놓고 문서 경로에
태우면 그 뒤 처리가 전부 따라온다.

관찰된 거스러미 두 가지 — 매뉴얼에 적어 둘 것.

- 카멜케이스 키가 공백으로 갈라진다(`payReasons` → `pay Reasons`). 값이 아니라 라벨이라
  검색에는 영향이 적지만, 싫으면 평문화 전에 키를 바꾼다.
- **`html_renderer` 로 넘길 함수가 toolbox 에 없다.** `tb.render_table` 은 시그니처가
  `(grid, num_cols)` 라 `html_renderer(str)->str` 계약에 안 맞고, 실제로 넣어야 하는 것은
  `enrichment.tabular_custom_fields.structural_html_renderer()` 다(toolbox 미수출).
  toolbox 주석은 "`html_renderer=` 로 표 렌더 주입" 이라고 안내하는데 **주입할 것이 없다.**
  → 09 반영 목록 ①

### B3. 단순화된 JSON 을 만드는 전처리

`json`(text_fields)이나 `json_mapping` 설정이 하나라도 매칭되는 doc_type 이면 `pre_source` 가
payload(dict/list)를 그대로 받으므로, 필요한 항목만 남긴 새 payload 를 돌려주면 된다
(대조군 테스트로 호출 확인). 매칭 설정이 없으면 위 B1 처럼 라우트가 필요하다.

### LLM 요청은 어디서 보내나 (요건에 포함된 항목)

- **정석은 설정이다** — `custom_field_*.yaml` 의 `extractor: llm`(프롬프트 파일·출력 필드),
  표 설명은 `enrichment.table_description`. 재시도·캐시·thinking 방언이 전부 붙어 있다.
- **facade 코드에서 부르는 것도 된다** — `async def post_parse` 로 바꿔 사내 API 를 호출한다
  (동기로 부르면 이벤트 루프가 막혀 같은 서버의 다른 문서까지 멈춘다).
- 다만 **toolbox 에 "설정된 LLM 을 부르는" 도우미가 없다.** 고객이 httpx 클라이언트·프롬프트
  로딩·재시도를 직접 짜게 된다. → 09 반영 목록 ②

---

## C. 마크다운 — 둘 다 이미 된다

같은 md 를 파싱·청킹해 확인했다.

```
# 상품 안내 / ## 개요 (본문 300자) / ## 금리표 (설명 문장 + md 표) / ## 유의사항 (본문 300자)
```

청크 3건이 나왔고 그중 1건이 `has_table=True` 인 **표 단독 청크**였다. 표 청크에는 그 절의
제목과 표 바로 위 문장이 함께 실린다(커밋 `ae2b5364`). C2 는 코드도 설정도 필요 없다 —
`chunking.table_as_chunk` 가 출고 기본 `true` 다.

C1(LaTeX)도 이미 처리된다. 파서가 `$$…$$` 블록을 `formula` 아이템으로 만들고
(docling md 백엔드), 청커가 `chunking/formula_text.py` 로 `$$` 구분자를 다시 씌워 싣는다.
실측 청크: `'$$E = mc^2$$'`. 인라인 `$…$` 는 본문 그대로 남는다.

**따라서 이 두 건에서 고객이 할 일은 "이미 된다는 것을 아는 것" 뿐이다.** 매뉴얼의
"이미 있는 기능부터" 목록(09 제안 파일 머리 주석)에 두 줄을 넣는다.

---

## D. HTML 섹션 헤더 지정 방식

내장 마커 승격은 문자 집합이 코드 상수(`_MARKER_CHARS = "◈◆◇▣■□❏●○▶▷"`)이고
doc_type 게이팅도 yaml 이다. `【 】` `<< >>` 처럼 부서마다 다른 기호는 대상이 아니다.

facade 코드로 푼다 — `.html/.htm/.md` 는 `pre_source` 가 **원문 문자열**을 받는다.

```python
MARKERS = (("【", "】", 2), ("■", "", 3))

def pre_source(self, ext, doc_type, data, work_dir=None, **kwargs):
    if ext not in (".html", ".htm"):
        return data
    def repl(m):
        body = m.group(1).strip()
        for open_ch, close_ch, level in MARKERS:
            if body.startswith(open_ch):
                return f"<h{level}>{body.strip(open_ch + close_ch).strip()}</h{level}>"
        return m.group(0)
    return re.sub(r"<p>(.*?)</p>", repl, data, flags=re.S)
```

실측 — `【1. 장비 개요】` `■ 1.1 주요 사양` `【2. 점검 절차】` 가 전부 `section_header` 로
잡혔고, 청킹하면 그 경계로 3청크가 나왔다. 새 부서 기호는 `MARKERS` 에 한 줄이다.

한 줄이 소제목인지 판정만 빌리고 싶으면 `tb.marker_heading_match(line)` 을 쓴다(내장 규칙과
같은 판정: 길이 상한 80, 서술형 종결 강등, 마커 반복 장식선 제외).

---

## E·F. 메타데이터 후처리와 추출

### E1·E2·F1. 문서 단위 — `post_parse` 한 곳

```python
DOC_NO = re.compile(r"문서번호:\s*(DOC-\d{4}-\d{4})")

def post_parse(self, ext, doc_type, result, **kwargs):
    doc = result.get("document") or {}
    body = "\n".join((t.get("text") or "") for t in doc.get("texts", []))
    found = DOC_NO.search(body)                       # F1: 본문에서 정규식 추출
    if found:
        tb.set_chunk_metadata(result, {                # E1: 값 변환은 yaml 과 같은 함수로
            "DOC_NO": found.group(1),
            "REV_DATE": tb.date_int_flex("2026-07-01"),
        })
    return result
```

실측 — 청크 3건 전부에 `DOC_NO=DOC-2026-0417`, `REV_DATE=20260701` 이 실렸다.

쓸 수 있는 변환기(전부 yaml 의 `transform:` 과 **같은 함수**):
`regex_sub` `regex_extract` `text_norm`(공백·문장부호 정규화) `to_int` `date_int`
`date_int_flex` `truncate` `html_text` `text` `json_to_markdown` `strip_inline_html`.
특수문자 제거는 `tb.sanitize`(제어문자·전각), 표현 정리는 `tb.tidy`.
사이트 전용 변환은 `tb.register_transform("won_to_int", fn)` 으로 등록하면 yaml 에서도 같은
이름으로 쓰인다 — **복합 후처리를 코드와 설정 양쪽에서 같은 파이프라인으로 굴리는 통로다.**

주의 두 가지.

- `result["metadata"]` 에 직접 쓰면 **파서 API 응답에만 보이고 청크에는 안 실린다.**
  청크까지 보내려면 `tb.set_chunk_metadata()` 다(문서는 KeyValueItem, 레코드는 각
  element 의 `metadata` 로 가른다). 문서 경로는 docling 왕복 비용이 있으니 쓸 때만 부른다.
- `post_parse` 에는 **file_path 가 안 들어온다.** 원본을 다시 읽어야 하는 후처리는
  라우트에서 하거나(A3), 09 의 `job` 이 생긴 뒤 `job.notes` 로 넘긴다. → 09 반영 목록 ③

### F2. 그 밖의 추출 방식

- 사내 API·코드표 조회: `async def post_parse` 로 바꾼다. 코드표 엑셀은 `tb.load_sheets`.
- 설정만으로 코드를 꽂는 자리도 있다 — `extractor: python`(`file`/`callable`). 계약은
  `def extract(text, document=None, doc_type=None, **kwargs) -> dict` 이고 코루틴도 된다.
  다만 **파이썬 파일이 config yaml 폴더 아래**여야 한다(임의 경로 실행 차단). facade 파일에
  두는 것과 성격이 다르므로, 값 추출이 목적이고 배포 단위를 설정 쪽에 두고 싶을 때만 쓴다.

### F3. 청크마다 다른 메타 — 훅 계약의 한계

`on_chunk` 는 **문자열 하나**만 돌려줄 수 있고, `info["metadata"]` 는 사본이라 거기에 써도
버려진다. 그래서 "이 청크에만 값을 붙인다" 는 `post_chunk` 에서 본문을 다시 보고 넣어야 한다.

```python
def post_chunk(self, rows, **kwargs):
    for r in rows:
        r.RISK = "high" if "손실" in r.text else "low"
    return rows
```

**09 설계의 `build_row(job, chunk, text)` 가 정확히 이 구멍을 메운다** — 청크와 본문을 함께
받는 자리가 생기므로 본문 재탐색이 사라진다.

---

## G. 최종 청크 결과 조작 — 전부 `post_chunk`

행은 `GenOSVectorMeta`(pydantic, `extra=allow`) 인스턴스다. pytest 로 세 레시피를 고정했다.

```python
# G1. 특정 청크 분리 — 필드가 딸린 사본을 만들고 본문만 갈아 끼운다
clone = v.model_copy(deep=True); clone.text = piece      # extra 필드도 따라온다

# G2. 작은 청크 결합
buf.text += "\n" + v.text

# G3·G4. 메타 일괄 변환과 제거
v.AMOUNT = int(tb.regex_sub(v.AMOUNT, pattern=r"\D", repl=""))
v.__pydantic_extra__.pop("INTERNAL_URL", None)           # 선언 필드는 None 으로

tb.refresh_stats(rows)          # 행을 늘리거나 줄였으면 필수(순번까지 다시 매긴다)
tb.refresh_stats(rows, reindex=False)   # 본문만 고쳤을 때
```

검증 결과: 3행 분리 후 `n_char [10,15,15]` · `i_chunk_on_doc [0,1,2]` · `n_chunk_of_doc 3`,
결합 후 `n_char 3`, 변환·제거 후 `model_dump()` 에 `AMOUNT=18000` 이고 `INTERNAL_URL` 없음.

되지만 거친 자리 — **분리·결합 도우미가 toolbox 에 없다.** 고객이 `model_copy` 와
`__pydantic_extra__` 같은 pydantic 내부를 알아야 한다. → 09 반영 목록 ④

설정으로 되는 대안도 매뉴얼에 함께 적는다(코드보다 먼저 확인할 것):
결합은 `chunking.chunk_mode: resize_all`, 분리는 레코드 경로의 `splittable`,
청크 버리기·치환은 `chunking.text_cleanup.rules` 의 `chunk`/`find`/`line`.

---

## H. 파싱 결과 후처리 · 청킹 입력 전처리

두 자리가 이미 있고 형태만 다르다.

| 하고 싶은 일 | 자리 | 받는 형태 |
|---|---|---|
| 파서가 만든 docling 문서를 손보기 | `post_parse` | `result["document"]` = dict |
| 청킹 직전 docling 문서를 손보기 | `pre_chunk`(kind `"docling"`) | dict |
| 파서 element 손보기 | `post_parse` | `result["elements"]` = list[dict] |
| 청킹 직전 element 손보기 | `pre_chunk`(kind `"parse"`) | list[dict] |

dict 를 직접 만지기 싫으면 왕복시킨다 — `DoclingDocument.model_validate(data)` 로 객체를
만들어 `iterate_items()` 로 고치고 `model_dump(mode="json")` 으로 되돌린다. 청커는 어차피
`model_validate` 로 복원하므로 결과는 같다.

**09 의 `on_document(job, doc)` 훅이 이 왕복을 없앤다** — 분석 직후·자동 설명 전의 문서
객체를 그대로 받는다. 지금은 그 시점에 끼어들 자리가 아예 없어서, "표 설명을 붙이기 전에
특정 표를 빼겠다" 같은 요구가 `post_parse`(이미 설명이 붙은 뒤)로 밀린다.

---

## 09 설계에 반영할 것

이번 검증에서 나온 것만 적는다. ①②④ 는 09 범위 밖의 별도 이슈 후보다.

| # | 내용 | 성격 |
|---|---|---|
| ① | toolbox 가 안내하는 `html_renderer` 에 넘길 함수가 toolbox 에 없다. `structural_html_renderer` 를 재수출하거나 주석을 고친다 | 결함(작음) |
| ② | facade 코드에서 LLM 을 부를 도우미가 없다. "LLM 은 yaml" 을 매뉴얼에 못 박거나 얇은 호출기를 재수출한다 | 문서/기능 |
| ③ | `post_parse` 에 원본 경로가 없다. 09 의 `job` 에 `file_path` 를 **명시**한다(`job.source` 는 xlsx 에서 격자로 바뀌어 경로가 사라진다) | **09 설계** |
| ④ | 청크 행 분리·결합 도우미 부재(`model_copy`·`__pydantic_extra__` 노출). 공용 모듈에 두고 toolbox 로 재수출 | 기능 후보 |
| ⑤ | 무설정 `.json` 이 `pre_source` 를 지나지 않는다. 09 의 `route_json` 도 같은 폴스루 구조다 — 파일 머리 주석에 "JSON 을 코드로 다루려면 ROUTES 한 줄" 을 예시로 박는다 | **09 문서** |
| ⑥ | `document` 와 `elements` 는 공존 불가. 혼합 요구는 한 표현으로 통일해야 한다는 규칙을 명시한다 | **09 문서** |
| ⑦ | C1·C2(LaTeX·md 표 단독 청크)는 이미 된다. "이미 있는 기능부터" 목록에 두 줄 추가 | **09 문서** |

③⑤⑥⑦ 은 [09-proposed/](09-proposed/) 두 파일에 이미 반영했다(파서 288줄, 청커 241줄).
①②④ 는 그 예시 안에 `tb.html_to_text` · `tb.split_row` · `tb.merge_small_rows` ·
`tb.drop_fields` 라는 **아직 없는 이름**으로 들어가 있다 — 별도 이슈가 닫히면 그대로 동작한다.

09 제안 설계의 값을 확인한 것도 두 가지다.

- **`job.notes` 는 A3 가 근거다.** `pre_source` 가 떼어 낸 표 밖 본문을 `post_parse` 로
  넘길 통로가 없어서 지금은 파일을 두 번 읽는다.
- **`build_row(job, chunk, text)` 는 F3 이 근거다.** `on_chunk` 의 "문자열만 반환" 계약
  때문에 청크별 메타는 본문 재탐색으로 밀려 있다.

## 매뉴얼 이관 계획

`examples/facade_hooks/` 에 예시 파일로 옮기고 README 표에 줄을 더한다. 이번에 실행까지
확인한 것만 우선 옮긴다(각 파일은 그대로 CLI 로 돌아간다).

| 파일(안) | 케이스 | 붙일 자리 |
|---|---|---|
| `hooks_xlsx_prose.py` | A2·A3 | `pre_source` + `route_tabular` |
| `hooks_json_flatten_route.py` | B1·B2·B4 | `ROUTES` + 자체 라우트 |
| `hooks_html_markers.py` | D1 | `pre_source` |
| `hooks_meta_from_body.py` | E1·F1 | `post_parse` |
| `hooks_post_chunk_rows.py` | G1~G4 | `post_chunk` |

`facade_hooks.md`(473줄, 09 의 7단계에서 다시 쓸 문서)에는 위 20건의 판정 표를
"이 요건은 어디서 푸나" 색인으로 넣는다.
