import logging
import re
import copy
import difflib
from typing import Dict, List, Optional, Union
from pathlib import Path

from openai import OpenAI
from docling_core.types.doc import TextItem, DocItemLabel, SectionHeaderItem
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


class TocExtractionOptions(PdfPipelineOptions):
    """ToC 추출 파이프라인 옵션"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # LLM 설정
        self.openai_api_key: Optional[str] = None
        self.openai_base_url: Optional[str] = "https://openrouter.ai/api/v1"
        self.model_name: str = "google/gemma-3-27b-it:free"
        self.temperature: float = 0.0
        self.top_p: float = 0
        self.seed: int = 33
        
        # ToC 추출 설정
        self.auto_extract_toc: bool = False
        self.convert_headers_to_text: bool = True
        
        # 구조 개선 설정
        self.improve_document_structure: bool = True  # ToC 기반 구조 개선
        self.similarity_threshold: float = 0.7  # 유사도 임계값 (difflib 기반으로 변경)


class TocExtractionPipeline(StandardPdfPipeline):
    """LLM을 사용한 ToC 추출 및 문서 구조 개선 파이프라인"""
    
    def __init__(self, pipeline_options: TocExtractionOptions):
        super().__init__(pipeline_options)
        self.toc_options: TocExtractionOptions = pipeline_options
        
        # OpenAI 클라이언트 초기화
        self.client = None
        if self.toc_options.openai_api_key:
            self.client = OpenAI(
                base_url=self.toc_options.openai_base_url,
                api_key=self.toc_options.openai_api_key,
            )
    
    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """문서 조립, ToC 추출 및 구조 개선"""
        # 기본 문서 조립 수행
        conv_res = super()._assemble_document(conv_res)
        
        # 자동 ToC 추출이 활성화된 경우
        if self.toc_options.auto_extract_toc and self.client:
            with TimeRecorder(conv_res, "toc_extraction_and_structure_improvement", scope=ProfilingScope.DOCUMENT):
                # 1. ToC 추출
                toc_content = self._extract_toc_with_llm(conv_res)
                
                # 2. ToC 기반 문서 구조 개선
                if toc_content and self.toc_options.improve_document_structure:
                    conv_res = self._improve_document_structure_with_toc(conv_res, toc_content)
                    _log.info("ToC 기반 문서 구조 개선 완료")
                
                # 3. ConversionResult에 ToC 정보 추가
                if not hasattr(conv_res, 'metadata') or conv_res.metadata is None:
                    conv_res.metadata = {}
                conv_res.metadata['table_of_contents'] = toc_content
                
                _log.info("LLM을 사용한 ToC 자동 추출 완료")
        
        return conv_res
    
    def clean_spaces(self, text: str) -> str:
        """텍스트 정리 - 공백 및 줄바꿈 정규화"""
        cleaned_lines = []
        for line in text.splitlines():
            cleaned_line = re.sub(r'\s+', ' ', line.strip())
            cleaned_lines.append(cleaned_line)
        return '\n'.join(cleaned_lines)
    
    def convert_all_section_headers_to_text(self, conv_res: ConversionResult) -> ConversionResult:
        """문서의 모든 SectionHeaderItem을 TextItem으로 변환"""
        if not self.toc_options.convert_headers_to_text:
            return conv_res
            
        new_texts = []
        
        for item in conv_res.document.texts:
            if isinstance(item, SectionHeaderItem):
                new_item = TextItem(
                    self_ref=item.self_ref,
                    parent=item.parent,
                    children=item.children,
                    content_layer=item.content_layer,
                    label=DocItemLabel.TEXT,
                    prov=item.prov,
                    orig=item.orig,
                    text=item.text,
                    formatting=item.formatting,
                    hyperlink=item.hyperlink
                )
                new_texts.append(new_item)
            else:
                new_texts.append(item)
        
        conv_res.document.texts = new_texts
        return conv_res
    
    def extract_raw_text(self, conv_res: ConversionResult) -> str:
        """ConversionResult에서 원시 텍스트 추출"""
        conv_res_copy = copy.deepcopy(conv_res)
        conv_res_copy = self.convert_all_section_headers_to_text(conv_res_copy)
        
        raw_texts = ""
        for text_item in conv_res_copy.document.texts:
            raw_texts += self.clean_spaces(text_item.text) + "\n"
        
        return raw_texts
    
    def _extract_toc_with_llm(self, conv_res: ConversionResult) -> str:
        """LLM을 사용하여 ToC 추출"""
        if not self.client:
            _log.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            return ""
        
        raw_text = self.extract_raw_text(conv_res)
        
        if not raw_text.strip():
            _log.warning("추출할 텍스트가 없습니다.")
            return ""
        
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a professional assistant trained to generate structured, hierarchical tables of contents "
                            "from Korean policy or research documents. Your job is to extract section titles and subtitles, "
                            "and organize them using Arabic numerals (e.g., 1, 1.1, 1.2). "
                            "Do not include page numbers. "
                            "Do not add explanations or comments. Only output a clean list of the table of contents using information from the document."
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "다음은 PDF 문서에서 추출한 텍스트입니다. 이 문서의 구조에 맞게 목차를 생성해주세요. "
                            "불필요한 설명 없이 순번과 제목만 출력해 주세요. 페이지 번호는 생략하고, 계층 구조는 숫자로 표현해 주세요.\n\n"
                            "문서 텍스트:\n\n"
                            f"{raw_text}"
                        )
                    }
                ]
            }
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.toc_options.model_name,
                messages=messages,
                temperature=self.toc_options.temperature,
                top_p=self.toc_options.top_p,
                seed=self.toc_options.seed
            )
            
            toc_content = completion.choices[0].message.content
            _log.info(f"LLM ToC 추출 성공: {len(toc_content)} 문자")
            return toc_content
            
        except Exception as e:
            _log.error(f"LLM ToC 추출 중 오류 발생: {str(e)}")
            return ""
    
    def apply_toc_headers(self, conv_res: ConversionResult, toc_content: str, threshold: float = None) -> ConversionResult:
        """
        개별 문서에 목차 기반 헤더를 적용하는 함수
        
        Args:
            conv_res: docling ConversionResult 객체
            toc_content: LLM이 생성한 목차 텍스트
            threshold: 텍스트 매칭 임계값 (None이면 옵션의 similarity_threshold 사용)
        
        Returns:
            수정된 conv_res
        """
        if threshold is None:
            threshold = self.toc_options.similarity_threshold
            
        # 1단계 목차 항목 추출
        top_level_toc = re.findall(r'^\d+\.\s*(.+?)(?=\n|$)', toc_content, re.MULTILINE)
        
        converted_indices = set()
        
        # 텍스트 아이템들 준비
        text_items = [
            (i, item.text.strip()) 
            for i, item in enumerate(conv_res.document.texts)
            if (isinstance(item, TextItem) and 
                item.label == DocItemLabel.TEXT and 
                len(item.text.strip()) >= 2)
        ]
        
        # 각 목차 항목을 문서의 텍스트와 매칭
        for toc_item in top_level_toc:
            toc_clean = toc_item.strip()
            if len(toc_clean) < 2:
                continue
            
            # difflib을 사용한 유사 텍스트 찾기
            text_only = [text for _, text in text_items]
            close_matches = difflib.get_close_matches(
                toc_clean, text_only, n=3, cutoff=0.4
            )
            
            if close_matches:
                # 가장 유사한 매치의 인덱스 찾기
                best_match_text = close_matches[0]
                best_match_idx = next(
                    (idx for idx, text in text_items if text == best_match_text), 
                    None
                )
                
                if best_match_idx is not None and best_match_idx not in converted_indices:
                    # 정확한 유사도 계산
                    similarity = difflib.SequenceMatcher(
                        None, toc_clean.lower(), best_match_text.lower()
                    ).ratio()
                    
                    if similarity >= threshold:
                        # TextItem을 SectionHeaderItem으로 변환
                        original_item = conv_res.document.texts[best_match_idx]
                        section_header = SectionHeaderItem(
                            self_ref=original_item.self_ref,
                            parent=original_item.parent,
                            children=original_item.children,
                            content_layer=original_item.content_layer,
                            prov=original_item.prov,
                            orig=original_item.orig,
                            text=original_item.text,
                            formatting=original_item.formatting,
                            hyperlink=original_item.hyperlink,
                            level=1
                        )
                        conv_res.document.texts[best_match_idx] = section_header
                        converted_indices.add(best_match_idx)
                        
                        _log.debug(f"'{toc_clean}' → '{best_match_text}' (유사도: {similarity:.3f}) 변환됨")
        
        _log.info(f"총 {len(converted_indices)}개 항목을 SectionHeaderItem으로 변환했습니다.")
        return conv_res
    
    def _improve_document_structure_with_toc(self, conv_res: ConversionResult, toc_content: str) -> ConversionResult:
        """
        ToC 기반으로 문서 구조 개선 - apply_toc_headers 사용
        """
        if not toc_content:
            return conv_res
        
        _log.info("apply_toc_headers를 사용하여 문서 구조 개선 시작")
        return self.apply_toc_headers(conv_res, toc_content)
    
    def extract_toc_from_conversion_result(self, conv_res: ConversionResult, improve_structure: bool = False) -> str:
        """외부에서 ConversionResult를 받아서 ToC 추출 및 선택적 구조 개선"""
        # 메타데이터에 이미 ToC가 있는지 확인
        if (hasattr(conv_res, 'metadata') and 
            conv_res.metadata and 
            'table_of_contents' in conv_res.metadata):
            toc_content = conv_res.metadata['table_of_contents']
            _log.info("기존 메타데이터에서 ToC를 찾았습니다.")
        else:
            # 없으면 새로 추출
            toc_content = self._extract_toc_with_llm(conv_res)
        
        # 구조 개선 수행 (요청된 경우)
        if improve_structure and toc_content and self.toc_options.improve_document_structure:
            conv_res = self.apply_toc_headers(conv_res, toc_content)
            _log.info("문서 구조 개선이 완료되었습니다.")
        
        return toc_content


def extract_toc_from_document(
    conv_res: ConversionResult, 
    openai_api_key: str,
    openai_base_url: str = "https://openrouter.ai/api/v1",
    model_name: str = "google/gemma-3-27b-it:free",
    improve_structure: bool = False
) -> str:
    """독립적인 함수로 ConversionResult에서 LLM을 사용해 ToC 추출 및 구조 개선
    
    Args:
        conv_res: 변환된 문서 결과
        openai_api_key: OpenAI API 키
        openai_base_url: API 기본 URL
        model_name: 사용할 모델명
        improve_structure: True면 ToC 기반으로 문서 구조를 개선
    
    Returns:
        추출된 ToC 문자열
    """
    # 이미 ToC가 메타데이터에 있는지 확인
    if (hasattr(conv_res, 'metadata') and 
        conv_res.metadata and 
        'table_of_contents' in conv_res.metadata):
        toc_content = conv_res.metadata['table_of_contents']
        _log.info("기존 메타데이터에서 ToC를 찾았습니다.")
        return toc_content
    
    # 새로 추출
    options = TocExtractionOptions()
    options.openai_api_key = openai_api_key
    options.openai_base_url = openai_base_url
    options.model_name = model_name
    options.improve_document_structure = improve_structure
    
    pipeline = TocExtractionPipeline(options)
    return pipeline.extract_toc_from_conversion_result(conv_res, improve_structure=improve_structure)


def extract_toc_and_improve_structure(
    conv_res: ConversionResult,
    openai_api_key: str,
    openai_base_url: str = "https://openrouter.ai/api/v1",
    model_name: str = "google/gemma-3-27b-it:free",
    similarity_threshold: float = 0.7
) -> tuple[str, ConversionResult]:
    """ToC 추출과 문서 구조 개선을 함께 수행하는 편의 함수
    
    Returns:
        (toc_content, improved_conv_res): 추출된 ToC와 개선된 ConversionResult
    """
    options = TocExtractionOptions()
    options.openai_api_key = openai_api_key
    options.openai_base_url = openai_base_url
    options.model_name = model_name
    options.improve_document_structure = True
    options.similarity_threshold = similarity_threshold
    
    pipeline = TocExtractionPipeline(options)
    toc_content = pipeline.extract_toc_from_conversion_result(conv_res, improve_structure=True)
    
    return toc_content, conv_res


# 사용 예시
if __name__ == "__main__":
    """
    사용 예시:
    
    # 방법 1: 파이프라인에서 자동 ToC 추출 + 구조 개선
    options = TocExtractionOptions()
    options.openai_api_key = "your-api-key"
    options.auto_extract_toc = True
    options.improve_document_structure = True
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: options}
    )
    result = converter.convert("document.pdf")
    
    # 방법 2: 기존 ConversionResult에서 ToC 추출만
    toc = extract_toc_from_document(result, openai_api_key="your-key")
    
    # 방법 3: ToC 추출 + 문서 구조 개선
    toc, improved_result = extract_toc_and_improve_structure(
        result, openai_api_key="your-key"
    )
    """
    pass 