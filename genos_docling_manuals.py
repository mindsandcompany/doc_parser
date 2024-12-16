import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional

import fitz
from fastapi import Request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.document import ConversionStatus
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    OcrEngine,
    PdfBackend,
    PdfPipelineOptions,
    TableFormerMode,
)
# docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption


class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow'

    text: str = None
    n_char: int = None
    n_word: int = None
    n_line: int = None
    i_page: int = None
    i_chunk_on_page: int = None
    n_chunk_of_page: int = None
    i_chunk_on_doc: int = None
    n_chunk_of_doc: int = None
    n_page: int = None
    reg_date: str = None
    bboxes: str = None


class DocumentProcessor:
    def __init__(self):
        self.page_chunk_counts = defaultdict(int)

    def load_documents_with_docling(self, file_path: str, **kwargs: dict) -> list[Document]:
        # docling 설정
        # 필요에 따라 사용자 지정 가능. 여기서는 genos_vanilla 와 비슷하게 PDF를 처리한다 가정.
        # TODO: kwargs 와의 연결
        # TODO: Langchain document 를 꼭 써야하나?
        from_formats = None  # 모든 지원 형식
        pdf_backend = PdfBackend.DLPARSE_V2
        artifacts_path = None
        ocr = True
        force_ocr = False
        ocr_engine = OcrEngine.EASYOCR
        ocr_options = EasyOcrOptions(force_full_page_ocr=force_ocr)
        device = AcceleratorDevice.AUTO
        num_threads = 4
        table_mode = TableFormerMode.FAST

        # pdf pipeline options 설정
        accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)
        pipeline_options = PdfPipelineOptions(
            accelerator_options=accelerator_options,
            do_ocr=ocr,
            ocr_options=ocr_options,
            do_table_structure=True,
            document_timeout=None
        )
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = table_mode

        if pdf_backend == PdfBackend.DLPARSE_V1:
            backend = DoclingParseDocumentBackend
        elif pdf_backend == PdfBackend.DLPARSE_V2:
            backend = DoclingParseV2DocumentBackend
        elif pdf_backend == PdfBackend.PYPDFIUM2:
            backend = PyPdfiumDocumentBackend
        else:
            backend = DoclingParseV2DocumentBackend

        pdf_format_option = PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=backend,
        )
        format_options = {
            # PDF, IMAGE 등 docling이 처리 가능한 입력 형식에 대해 옵션 할당
            # TODO: option 정리할 것
        }

        # from_formats를 None으로 두면 docling이 모든 형식을 허용.
        doc_converter = DocumentConverter(
            allowed_formats=from_formats,
            format_options=format_options,
        )

        # 실제 변환 실행
        # ConversionResult 리스트를 받는다.
        conv_results = doc_converter.convert_all([file_path], raises_on_error=True)

        documents = []
        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                # docling 문서 구조에서 text 파싱
                # conv_res.document 는 docling-core 의 Document 객체
                # 이 문서에서 page 단위로 text 추출. docling의 Document는 pages 단위로 구성됨
                # conv_res.document.pages : list of Page
                # 각 Page의 text 를 추출하여 langchain Document를 만든다.
                for i, page in enumerate(conv_res.document.pages):
                    # page.text : 이 페이지의 text
                    # metadata에 page 정보 담기
                    doc = Document(page_content=page.text, metadata={"page": i})
                    documents.append(doc)
            else:
                raise Exception(f"Failed to load document: {conv_res.input.file}")

        return documents

    def load_documents(self, file_path: str, **kwargs: dict) -> list[Document]:
        # ducling 방식으로 문서 로드
        documents = self.load_documents_with_docling(file_path, **kwargs)
        return documents

    def split_documents(self, documents, **kwargs: dict) -> list[Document]:
        ##FIXME: Hierarchical Chunker 로 수정
        text_splitter = RecursiveCharacterTextSplitter(**kwargs)
        chunks = text_splitter.split_documents(documents)
        chunks = [chunk for chunk in chunks if chunk.page_content]
        if not chunks:
            raise Exception('Empty document')

        for chunk in chunks:
            self.page_chunk_counts[chunk.metadata['page']] += 1
        return chunks

    def compose_vectors(self, chunks: list[Document], file_path: str, **kwargs: dict) -> list[dict]:
        pdf_path = file_path.replace('.hwp', '.pdf').replace('.txt', '.pdf').replace('.json', '.pdf')
        if os.path.exists(pdf_path):
            doc = fitz.open(pdf_path)

        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=max([chunk.metadata['page'] for chunk in chunks]) if chunks else 0,
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z'
        )

        current_page = None
        chunk_index_on_page = 0
        vectors = []
        for chunk_idx, chunk in enumerate(chunks):
            page = chunk.metadata['page']
            text = chunk.page_content

            if page != current_page:
                current_page = page
                chunk_index_on_page = 0

            bboxes_json = None
            if os.path.exists(pdf_path):
                fitz_page = doc.load_page(page)
                bboxes = []
                # text 검색 시 fitz의 search_for 문맥이 주어진 text chunk 에 매칭되는 바운딩박스를 찾을 수 있는지 확인
                # 많은 경우 chunk가 PDF 내 같은 text를 그대로 match하지 못할 수 있음. 
                # 여기서는 원본 genos_vanilla와 동일한 로직 유지.
                # 특정 성능 문제나 결과 없을 경우 try-except 추가 가능.
                search_results = fitz_page.search_for(text)
                for rect in search_results:
                    bboxes.append({
                        'p1': {'x': rect[0] / fitz_page.rect.width, 'y': rect[1] / fitz_page.rect.height},
                        'p2': {'x': rect[2] / fitz_page.rect.width, 'y': rect[3] / fitz_page.rect.height},
                    })
                bboxes_json = json.dumps(bboxes)

            vectors.append(GenOSVectorMeta.model_validate({
                'text': text,
                'n_char': len(text),
                'n_word': len(text.split()),
                'n_line': len(text.splitlines()),
                'i_page': page,
                'i_chunk_on_page': chunk_index_on_page,
                'n_chunk_of_page': self.page_chunk_counts[page],
                'i_chunk_on_doc': chunk_idx,
                'bboxes': bboxes_json,
                **global_metadata
            }))
            chunk_index_on_page += 1

        return vectors

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        documents: list[Document] = self.load_documents(file_path, **kwargs)
        await assert_cancelled(request)

        chunks: list[Document] = self.split_documents(documents, **kwargs)
        await assert_cancelled(request)

        vectors: list[dict] = self.compose_vectors(chunks, file_path, **kwargs)
        return vectors


class GenosServiceException(Exception):
    # GenOS 와의 의존성 부분 제거를 위해 추가
    def __init__(self, error_code: str, error_msg: Optional[str] = None, msg_params: Optional[dict] = None) -> None:
        self.code = 1
        self.error_code = error_code
        self.error_msg = error_msg or "GenOS Service Exception"
        self.msg_params = msg_params or {}

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(code={self.code!r}, errMsg={self.error_msg!r})"


# GenOS 와의 의존성 제거를 위해 추가
async def assert_cancelled(request: Request):
    if await request.is_disconnected():
        raise GenosServiceException(1, f"Cancelled")
