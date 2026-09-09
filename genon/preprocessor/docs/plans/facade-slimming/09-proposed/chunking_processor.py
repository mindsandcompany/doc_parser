# 청킹용 전처리기
#
# 파서 결과를 받아 청킹한다. 원본문서를 읽거나 분석하지 않는다.
#
# 처리 순서. 아래 __call__ 에서 메소드호출 순서와 동일
#   파서 결과 -> 준비 -> pre_chunk -> 자르기 -> 행 만들기 -> post_chunk -> DB 행 목록
#   청크 한 건이 만들어질 때마다 on_chunk 를 지난다.
#
# 목차
#   1부 흐름
#   2부 문서 종류별 설정       청크 크기 등을 문서마다 다르게 할 때. 코드보다 여기가 먼저다
#   3부 사용자 수정가능 메소드   훅 3개
#   4부 단계 교체
#
# 단독 실행: python chunking_processor.py parsed.json -o chunks.json
from pydantic import BaseModel

from genon.preprocessor.facade.chunking import smart_chunker as sc
from genon.preprocessor.facade.core import toolbox as tb
from genon.preprocessor.facade.core.chunker import ChunkerCore
from genon.preprocessor.facade.core.errors import GenosServiceException


class GenOSVectorMeta(BaseModel):
    """청크정보 구성.

    메타데이터를 추가하려면 필드를 추가하고 값은 build_row() 에서 채운다.
    extra="allow" 선언에 없는 필드도 저장되지만, 가독성을 위해서 이곳에 명시하는 것을 권장한다.
    """

    class Config:
        extra = "allow"

    text: str = None
    # 기존 필드 그대로 (n_char, i_page, title, appendix, has_table 등)


class GenosSmartChunker(sc.SmartChunkerBase):
    """청킹 옵션. 실제 청킹 코드는 facade/chunking/smart_chunker.py 에 있다."""

    PICTURE_ANNOTATION_TEXT = True          # 그림 설명을 청크 본문에 함께 싣는다
    TABLE_DESCRIPTION_MODE = "prefix_only"  # 표 설명은 표 청크 앞에만. full 이면 본문에도
    CHUNK_HEADER_PREFIX = "HEADER: "        # 청크 앞 라벨. 빈 문자열이면 경로만 붙는다
    CHUNK_HEADER_SEP = " > "                # 상위 제목과 하위 제목 사이
    CHUNK_PATH_SEP = " | "                  # 같은 단계의 제목이 여러 개일 때
    CHUNK_PATH_MAX_LEAVES = 5               # 경로가 이보다 많으면 뒤를 줄인다


class DocumentProcessor(ChunkerCore):
    """청킹 전용 전처리기. main.py 가 /chunker 요청을 이 클래스로 보낸다."""

    IS_CHUNKER = True                 # 이 표시가 있어야 /chunker API 가 열린다
    VECTOR_META = GenOSVectorMeta
    CHUNKER = GenosSmartChunker

    # 자르지 않고 한 건을 청크 하나로 처리할 element category. 엑셀 행과 JSON 레코드가
    # 그렇다. 우리 파서가 다른 이름을 쓰면 여기에 더한다. 예: | {"my_row"}
    ROW_CATEGORIES = ChunkerCore.ROW_CATEGORIES

    # --- 1부. 흐름 ---

    async def __call__(self, request, file_path="", **kwargs):
        """ 청킹 진입점
        job 은 파서 파일의 job 과 다른 객체이며 필드도 다르다.
            job.kind      "docling" 은 문서에서 온 결과, "parse" 는 행이나 레코드
            job.data      파서 결과        job.doc_type  문서 종류
            job.metadata  문서 단위 정보    job.params    요청 파라미터
            job.config    적용된 설정(2부)  job.notes     단계 사이 값 전달용 dict

        훅에서는 job 을 kwargs["job"] 으로 꺼낸다.
        job.kind 와 on_chunk 의 info["kind"] 는 값이 다르다.
            "docling" -> "docling",  "parse" -> "row"(행, 레코드) 또는 "text"(그 밖)
        """
        job = self.start_job(request, file_path, **kwargs)  # 입력 판별, 문서별 설정 적용
        job.data = await self.run_pre_chunk(job)            # pre_chunk() 호출
        chunks = await self.split(job)                      # 자르기
        rows = await self.build_rows(job, chunks)           # 청크를 DB 행으로
        return await self.run_post_chunk(job, rows)         # post_chunk() 호출

    async def split(self, job):
        """
        청킹 방식 구분
        - 일반 서술형 문서(docling): GenosSmartChunker를 사용한 청킹
        - 행, 레코드: 공통 분할기로 청킹
        """
        if job.kind == "docling":
            return await self.split_document(job)
        return await self.split_records(job)

    async def build_rows(self, job, chunks):
        """청크를 DB 행 목록으로 바꾼다.
        """
        rows = []
        for chunk in chunks:
            text = self.build_text(job, chunk)                 # 접두 + 제목 경로 + 본문
            text = await self.run_on_chunk(job, chunk, text)   # on_chunk() 호출
            if text is None:
                continue                                       # tb.DROP 이면 이 청크는 버린다
            text = self.mask_sensitive(job, text)              # 개인정보 라벨과 마스킹
            text = self.clean_text(job, text)                  # 설정의 text_cleanup 적용
            rows.append(self.build_row(job, chunk, text))
        self.number_rows(job, rows)                            # 순번과 통계
        return rows

    def build_text(self, job, chunk):
        """청크 한 건의 본문을 조립한다.

        문서를 알아볼 수 있는 값(카드 이름, 문의 유형 등)이 제목 경로보다 앞에 온다.
        """
        return self.doc_prefix(job, chunk) + self.header_line(job, chunk) + chunk.text

    def build_row(self, job, chunk, text):
        """청크 하나를 DB 행 하나로 만든다.

        청크 메타데이터를 추가한다. on_chunk 는 본문 문자열만
        돌려줄 수 있고, 거기서 info["metadata"] 를 고쳐도 사본이라 반영되지 않는다.
            .set_extra(RISK="high" if "손실" in text else "low")

        여기 들어오는 text 는 접두와 제목 경로가 붙고 정리까지 끝난 값이다.
        순수 본문으로 계산하려면 chunk.text 를 쓴다.
        """
        return (self.row_builder(job)
                .set_text(text)
                .set_page_info(chunk)          # 페이지 번호와 그 페이지 안에서의 순번
                .set_chunk_bboxes(chunk)       # 원본에서의 위치 좌표
                .set_media_files(chunk)        # 표와 이미지 파일 경로
                .set_table_info(chunk)         # 표 청크 여부와 표가 나뉜 조각 번호
                .set_metadata(job.metadata)    # 문서 단위 정보. 모든 청크에 같은 값
                .build(self.VECTOR_META))

    # --- 2부. 문서 종류별 설정 ---
    #
    # 설정 파일(chunking_processor_config.yaml)은 모든 문서에 공통으로 적용된다.
    # 문서 종류마다 다르게 하려면 아래 표에 적는다. 키는 설정 파일의 경로를 점으로 이어 쓴다.
    #
    # 옵션 적용 순서(뒤로 갈수록 우선): 설정 파일, CONFIG_BY_DOC_TYPE, config_for(), 요청이 보낸 값
    # 적용된 값은 job.config 와 결과에 남는다.

    CONFIG_BY_DOC_TYPE = {
        # "faq":    {"chunking.chunk_size": 500,            # 한 문답이 짧다
        #            "chunking.min_chunk_size": 0},         # 0 이면 하한 보정을 하지 않는다
        # "manual": {"chunking.chunk_mode": "split_only",   # 절 단위를 유지한다
        #            "chunking.chunk_size": 2000},
    }

    def config_for(self, job):
        """[사용자 수정가능 0] 위 표로 안 되는 경우. 문서 내용이나 요청값으로 정한다.

        위 표와 같은 모양의 dict 를 돌려주고, 빈 dict 면 아무것도 바뀌지 않는다.

            if job.metadata.get("GROUP_C") == "INS":
                return {"guardrail.masking_enabled": True}
        """
        return {}

    # --- 3부. 사용자 수정가능 메소드 ---
    #
    # 공통 규칙은 파서와 동일
    #   1. 요청 값이 필요하면 **kwargs 를 붙인다. job 은 kwargs["job"] 으로 꺼낸다.
    #   2. 외부 API 를 부르려면 async def 로 바꾼다.
    #   3. self 에 값을 저장하지 않는다. 단계 사이 전달은 job.notes 를 쓴다.
    #   4. 실패는 GenosServiceException 으로 알린다. 일부만 실패한 경우라면 그 건만
    #      건너뛰고 post_chunk 에서 결과에 남긴다.

    def pre_chunk(self, kind, data, **kwargs):
        """[사용자 수정가능 1] 청킹하기 전 전처리

            kind == "parse"    data 는 list[dict] (엑셀 행, JSON 레코드)
            kind == "docling"  data 는 dict. 문서 객체가 아니라 JSON 으로 펼친 형태라
                               본문은 data["texts"][i]["text"] 로 꺼낸다
        """
        return data

    def on_chunk(self, text, info, **kwargs):
        """[사용자 수정가능 2] 청크 한 건이 만들어진 직후 호출.

        돌려주는 값은 셋 중 하나다.
            문자열   그 문자열이 이 청크의 본문이 된다
            None     아무것도 하지 않는다. return 을 빠뜨려도 청크가 사라지지 않는다
            tb.DROP  이 청크를 저장하지 않는다. 순번은 다시 매겨진다

        info 구조
            kind      "docling"(문서), "row"(행, 레코드), "text"(그 밖)
            page      페이지 번호(1부터)    index  지금까지의 순번(참고용)
            headings  제목 경로. 문서에서 온 청크만 채워진다
            metadata  문서나 레코드 정보. 사본이라 여기에 써도 저장되지 않는다

            if "상담직원용" in text:
                return tb.DROP
            return text.replace("[내부]", "")
        """
        return None

    def post_chunk(self, rows, **kwargs):
        """[사용자 수정가능 3] 청킹 결과 후처리.

        여기서 가능한 작업
        - 청크를 쪼개거나 합치는 것
        - 값을 일괄 변환하거나 지우는 것

            rows = tb.split_row(rows, when=lambda r: len(r.text) > 4000)
            rows = tb.merge_small_rows(rows, min_chars=80)
            for r in rows:
                r.AMOUNT = tb.to_int(tb.regex_sub(r.AMOUNT, pattern=r"\\D", repl=""))
                tb.drop_fields(r, "INTERNAL_URL")
            tb.refresh_stats(rows)

        본문을 고쳤거나 행을 지웠으면 통계를 다시 계산한다.
        - 행 수가 바뀌었으면 tb.refresh_stats(rows)
        - 본문만 고쳤으면 tb.refresh_stats(rows, reindex=False)

        참고: 본문만 고치거나 청크를 버리는 일은 on_chunk가 적합. 통계가 자동적용.
        """
        return rows

    # --- 4부. 단계 교체 ---
    #
    # 만드는 방식 자체를 바꾸려면 1부의 함수를 같은
    # 이름으로 다시 정의한다. 크기나 자르는 방식만 바꾸려면 2부 설정이 먼저다.
    #
    #   def build_text(self, job, chunk):
    #       return f"[{job.metadata.get('title', '')}] " + chunk.text


if __name__ == "__main__":
    DocumentProcessor.cli()
