# 파싱용 전처리기
#
# 파일 하나를 받아 파싱 후 JSON 으로 돌려준다. 청킹은 하지 않는다.
#
# 처리 순서. 아래 __call__ 에서 메소드호출 순서와 동일
#   파일 -> 준비 -> pre_source -> 확장자별 처리 -> post_parse -> JSON
#   문서를 만드는 처리는 중간에 on_document 를 지난다.
#
# 읽는 순서
#   1부 흐름
#   2부 확장자별 처리     새 확장자는 ROUTES 에 한 줄
#   3부 문서 종류별 설정  기능을 켜고 끈다. 코드보다 여기가 먼저다
#   4부 사용자 수정가능 메소드      훅 3개
#   5부 단계 교체         원본기능 자체를 바꿀 때
#
# 결과의 형태는 둘 중 하나로 구성된다.
#   문서형 {"document": {...}}   pdf hwp docx ppt md html, 설정을 갖춘 json 과 엑셀
#   행형   {"elements": [...]}   엑셀 행, JSON 레코드, 오디오, 그 밖
# 한 응답에 둘을 담아도 청커는 document 만 본다. 표는 행으로 본문은 문단으로 내보내려면
# 한쪽 표현으로 통일한다.
#
# 단독 실행 예시: python parser_processor.py 계약서.pdf --doc-type contract -o parsed.json
from genon.preprocessor.facade.core import toolbox as tb
from genon.preprocessor.facade.core.errors import GenosServiceException
from genon.preprocessor.facade.core.parser import ParserCore


class DocumentProcessor(ParserCore):
    """파싱 전용 전처리기. main.py 가 /parser 요청을 이 클래스로 보낸다."""

    IS_PARSER = True   # 이 표시가 있어야 /parser API 가 열린다

    # --- 1부. 흐름 ---

    async def __call__(self, request, file_path, **kwargs):
        """파서 진입점

        job 은 요청의 정보를 담은 값 객체다.
            job.ext        확장자(소문자)      job.doc_type  문서 종류(소문자)
            job.file_path  원본 파일 경로      job.source    지금 처리할 입력
            job.params     요청 파라미터       job.config    적용된 설정(3부)
            job.work_dir   임시 폴더(자동 삭제) job.notes     단계 사이 값 전달용 dict

        job.source 는 확장자에 따라 경로가 아니라 이미 읽어 둔 데이터일 수 있다. 엑셀은
        셀 값 표, json 은 dict, md 와 html 은 문자열이다. 원본 경로는 job.file_path 를 쓴다.
        훅에서는 job 을 kwargs["job"] 으로 꺼낸다. 훅 인자 목록에는 없다.
        """
        job = self.start_job(request, file_path, **kwargs)  # 확장자 확인, 문서별 설정 적용
        job.source = await self.run_pre_source(job)         # pre_source() 호출
        result = await self.run_route(job)                  # ROUTES 에서 골라 실행
        return await self.run_post_parse(job, result)       # post_parse() 호출

    async def document_to_response(self, job, doc):
        """문서를 응답 JSON 으로 만든다. 문서를 만드는 라우트 5개가 공유한다.

        메소드 호출 순서 유지 필수
        """
        doc = self.on_document(job, doc)      # LLM 설명 붙이기 전 (사용자 수정가능 2)
        doc = await self.enrich(job, doc)     # 표 설명, 이미지 설명, 항목 추출
        return self.build_response(job, doc)  # {"document": ..., "metadata": ...}

    async def records_to_response(self, job, rows):
        """엑셀 행이나 JSON 레코드 목록을 응답 JSON 으로 만든다.

        rows 한 건이 청크 하나가 된다. 키는 셋이다.
            content   검색에 걸리는 본문
            metadata  적재 DB 컬럼이 되는 값. 검색 본문에는 들어가지 않는다
            category  청커가 이 이름만 보고 처리 방식을 정한다. ROW_CATEGORIES 에 있는
                      이름(tabular_row, custom_fields_row)이면 자르지 않고 한 건을 청크
                      하나로 만들고, 그 밖이면 크기에 맞춰 자른다
        id, page, coordinates 는 tb.make_elements() 가 채운다.
        """
        return await self.describe_tables(job, rows)   # 행 안에 표가 있으면 설명을 붙인다

    # --- 2부. 확장자별 처리 ---
    #
    # 확장자에 해당되는 함수를 호출.
    # 위쪽에 정의된 부분부터 처리하며, 그 함수가 None 을 돌려주면 다음 줄로
    # 넘어간다. 마지막 줄(None)은 나머지 전부를 받으므로 항상 맨 아래에 둔다.
    #
    # 새 확장자 추가방법은 아래와 같다.
    # 예를 들어 .tsv 를 표로 다루려면
    # ((".tsv",), "route_tabular") 를 맨 위에 넣고 pre_source 에서 표 형태로 바꾼다.
    #
    # json 은 예외다. 아래 route_json 은 custom_fields 설정이 매칭될 때만 동작한다.
    # 매칭이 없으면 파일을 읽지도 않고 넘긴다. 설정 없이 json 을 코드로 다루려면 pre_source 가
    # 아니라 이 표에 정보를 추가해야 한다.
    #
    #   ROUTES = (((".json",), "route_json_ours"),) + DocumentProcessor.ROUTES
    #
    #   async def route_json_ours(self, job):
    #       if job.doc_type != "ins_api":
    #           return None                                  # 다른 문서는 원래 경로로
    #       picked = [x for x in job.source["items"] if x["type"] == "product"]
    #       md = tb.json_to_markdown(picked, html_renderer=tb.html_to_text())
    #       doc = self.parse_document(job, md, ext=".md")     # 마크다운으로 보고 분석
    #       return await self.document_to_response(job, doc)

    ROUTES = (
        ((".csv", ".xlsx", ".xlsm"),       "route_tabular"),  # 표 파일
        ((".hwp", ".hwpx", ".hml"),        "route_hwp"),
        ((".docx",),                       "route_docx"),
        ((".pdf", ".html", ".htm", ".md"), "route_docling"),  # 배치 분석이 필요한 문서
        ((".json",),                       "route_json"),
        ((".ppt", ".pptx"),                "route_ppt"),
        (None,                             "route_other"),    # 나머지 전부. 항상 마지막
    )

    async def route_docling(self, job):
        """pdf, html, htm, md. 배치와 표를 분석해 문서 하나로 만든다."""
        if job.ext == ".md" and not self.md_uses_docling(job):
            return None                              # text 모드면 route_other 가 받는다
        markup = await self.prepare_markup(job)      # HTML 정리, 머리 정보 분리
        doc = self.parse_document(job, markup)       # 배치 분석, OCR, 표 구조 인식
        return await self.document_to_response(job, doc)

    async def route_tabular(self, job):
        """csv, xlsx, xlsm. 설정에 행 매핑이 있으면 행 단위로, 없으면 문서로 만든다."""
        sheets = await self.read_sheets(job)         # {시트명: 표}
        if self.has_row_mapping(job):
            return await self.records_to_response(job, self.map_rows(job, sheets))
        if self.xlsx_as_document(job):               # 표를 문서처럼 다루는 설정
            return await self.document_to_response(job, self.parse_sheets(job, sheets))
        return await self.records_to_response(job, self.sheets_to_rows(job, sheets))

    async def route_json(self, job):
        """json. 레코드 매핑이 먼저, 본문 항목 설정이 있으면 문서로, 둘 다 없으면 넘긴다."""
        if self.has_record_mapping(job):
            return await self.records_to_response(job, await self.map_records(job))
        if self.has_json_text_fields(job):
            return await self.document_to_response(job, await self.parse_json_text(job))
        return None                                  # route_other 가 받는다

    async def route_hwp(self, job):     # hwp, hwpx, hml
        return await self.document_to_response(job, self.parse_hwp(job))

    async def route_docx(self, job):    # docx
        return await self.document_to_response(job, self.parse_docx(job))

    async def route_ppt(self, job):
        """ppt, pptx. PDF 로 변환한 뒤 분석하고, 변환이 실패하면 텍스트만 뽑는다."""
        doc = self.parse_ppt(job)
        if doc is None:
            return self.parse_plain(job)
        return await self.document_to_response(job, doc)

    async def route_other(self, job):
        """나머지 전부. doc, txt, 이미지 등에서 텍스트만 뽑는다."""
        return self.parse_plain(job)

    # --- 3부. 문서 종류별 설정 ---
    #
    # 설정 파일(parser_processor_config.yaml)은 모든 문서에 공통으로 적용된다. 문서 종류마다
    # 다르게 하려면 아래 표에 적는다. 키는 설정 파일의 경로를 점으로 이어 쓰고, 설정 파일에
    # 있는 항목이면 무엇이든 된다.
    #
    # 옵션 적용 순서(뒤로 갈수록 우선): 설정 파일, CONFIG_BY_DOC_TYPE, config_for(), 요청이 보낸 값
    # 적용된 값은 job.config 와 결과에 남으므로 나중에 추적할 수 있다.

    CONFIG_BY_DOC_TYPE = {
        # "press":    {"enrichment.table_description.enable": False,   # 표가 없으니 낭비
        #              "chunking.chunk_size": 500},
        # "manual":   {"enrichment.image_description.enable": True},   # 그림 설명이 중요
        # "contract": {"ocr.ocr_mode": "force"},                       # 스캔본이 많다
    }

    def config_for(self, job):
        """[사용자 수정가능 0] 위 표로 안 되는 경우. 문서 내용이나 요청값으로 정한다.

        위 표와 같은 모양의 dict 를 돌려주고,
        빈 dict 면 아무것도 바뀌지 않는다.

            if job.params.get("dept") == "IR":
                return {"enrichment.doc_summary.enable": True}
        """
        return {}

    # --- 4부. 사용자 수정가능 메소드 ---
    #
    # 기본 동작은 받은 것을 그대로 돌려주는 것이라, 안 고치면
    # 결과가 달라지지 않는다.
    #
    # 공통 규칙
    #   1. 요청 값이 필요하면 마지막에 **kwargs 를 붙인다. job 도 여기로 들어온다.
    #   2. 외부 API 를 부르려면 async def 로 바꾼다. 동기 호출은 이 서버가 처리 중인
    #      다른 요청까지 멈춘다.
    #   3. self 에 값을 저장하지 않는다. 객체 하나가 모든 요청을 처리하므로 값이 섞인다.
    #      단계 사이 전달은 kwargs["job"].notes 를 쓴다. 클래스 상수는 괜찮다.
    #   4. 실패는 GenosServiceException 으로 알린다. 한 건이 실패해도 나머지를 살려야
    #      한다면 던지지 말고 건너뛴 뒤 결과에 남긴다.
    #        raise GenosServiceException("1", "건수가 맞지 않아 처리를 중단했습니다")

    def pre_source(self, ext, doc_type, data, work_dir=None, **kwargs):
        """[사용자 수정가능 1] 파싱 전에 입력 전처리.

        형식이 조금 달라 그대로는 처리가 안 될 때, 아는 형식으로 바꿔 주면 기존 처리
        함수가 그대로 받는다. 들어오는 형태는 확장자마다 다르고 같은 형태로 돌려준다.
            .json        dict 또는 list. 형식이 깨져 있으면 str
            .md .html    str (파일 내용 전체)
            .xlsx .csv   {시트명: [[셀, ...], ...]}. 시트를 행 단위로 편 셀 값 표이며
                         병합 셀은 각 칸에 같은 값이 이미 채워져 있다
            그 밖         str (파일 경로). 새 파일은 work_dir 안에 만든다

            if ext == ".xlsx" and doc_type == "branch_list":
                return {name: rows[2:] for name, rows in data.items()}   # 머리 두 줄 제거
        """
        return data

    def on_document(self, job, doc):
        """[사용자 수정가능 2] 파싱이 끝나고 LLM 이 설명을 붙이기 전이다.

        설명은 표 설명, 이미지 설명, 문서 요약, custom_fields 항목 추출을 말한다.
        LLM 호출 전에 구조를 고칠 때 쓴다.
        제목 레벨이 잘못 잡혔거나 특정 표를 설명 대상에서 제외시키는 경우가 해당된다.
        doc.texts, doc.tables, doc.iterate_items() 를 주로 수정할 수 있다.

            for item in doc.texts:
                if item.text.startswith("부칙"):
                    item.label = "section_header"
            return doc
        """
        return doc

    def post_parse(self, ext, doc_type, result, **kwargs):
        """[사용자 수정가능 3] 파싱결과 후처리.

        result 구조
            result["document"]  문서 경로 산출 dict. 본문은 ["texts"][i]["text"]
            result["elements"]  행 경로 산출 list[dict]. 한 건은 {"content", "metadata"}
            result["metadata"]  문서 한 건에 대한 정보

        청크에 실을 값은 tb.set_chunk_metadata() 로 넣는다. result["metadata"] 에 직접
        쓰면 이 API 응답에만 보이고 청크에는 실리지 않는다. 이렇게 넣는 값은 그 문서의
        모든 청크에 동일하게 붙는다. 청크마다 다른 값은 chunking_processor.py 의 build_row 다.

            async def post_parse(self, ext, doc_type, result, **kwargs):
                emp_no = (result.get("metadata") or {}).get("EMP_NO")
                if emp_no:
                    tb.set_chunk_metadata(result, {"DEPT_NM": await fetch_dept(emp_no)})
                return result

        일부가 실패해도 나머지를 살릴 때는 실패한 건만 빼고 그 사실을 결과에 남긴다.
        pre_source 가 kwargs["job"].notes 에 담아 둔 값을 여기서 꺼내 붙일 수도 있다.
        """
        return result

    # --- 5부. 단계 교체 ---
    #
    # 자동으로 수행되는 메소드를 바꾸려면 1부나 2부의 함수를
    # 같은 이름으로 다시 정의한다. 기능을 통째로 끄는 것이 목적이면 3부 설정을 우선 고려한다.
    #
    #   async def enrich(self, job, doc):            # 표 설명을 우리 기준으로 검사
    #       doc = await super().enrich(job, doc)     # 기본 동작을 먼저 실행
    #       for table in doc.tables:
    #           caption = (table.captions or [""])[0]
    #           if not all(k in caption for k in ("기준일", "연 %")):
    #               table.captions = []              # 기준 미달이면 설명만 뗀다
    #       return doc
    #
    # 2부의 route_* 도 같은 방법으로 바꾼다. super() 를 부르지 않으면 기본 동작은
    # 실행되지 않는다.


if __name__ == "__main__":
    DocumentProcessor.cli()
