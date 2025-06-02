import logging
import warnings
import json
import re
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union, cast, Dict, Any

# from docling_core.types import DoclingDocument
from docling_core.types.doc import BoundingBox, DocItem, ImageRef, PictureItem, TextItem
from docling_core.types.doc.document import DocTagsDocument
from PIL import Image as PILImage

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    ApiVlmOptions,
    HuggingFaceVlmOptions,
    InferenceFramework,
    ResponseFormat,
    VlmPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.models.api_vlm_model import ApiVlmModel
from docling.models.hf_mlx_model import HuggingFaceMlxModel
from docling.models.hf_vlm_model import HuggingFaceVlmModel
from docling.pipeline.base_pipeline import PaginatedPipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder
from openai import OpenAI

_log = logging.getLogger(__name__)


class VlmPipeline(PaginatedPipeline):

    def __init__(self, pipeline_options: VlmPipelineOptions):
        super().__init__(pipeline_options)
        self.keep_backend = True

        self.pipeline_options: VlmPipelineOptions

        artifacts_path: Optional[Path] = None
        if pipeline_options.artifacts_path is not None:
            artifacts_path = Path(pipeline_options.artifacts_path).expanduser()
        elif settings.artifacts_path is not None:
            artifacts_path = Path(settings.artifacts_path).expanduser()

        if artifacts_path is not None and not artifacts_path.is_dir():
            raise RuntimeError(
                f"The value of {artifacts_path=} is not valid. "
                "When defined, it must point to a folder containing all models required by the pipeline."
            )

        # force_backend_text = False - use text that is coming from VLM response
        # force_backend_text = True - get text from backend using bounding boxes predicted by SmolDocling doctags
        self.force_backend_text = (
            pipeline_options.force_backend_text
            and pipeline_options.vlm_options.response_format == ResponseFormat.DOCTAGS
        )

        self.keep_images = self.pipeline_options.generate_page_images

        if isinstance(pipeline_options.vlm_options, ApiVlmOptions):
            self.build_pipe = [
                ApiVlmModel(
                    enabled=True,  # must be always enabled for this pipeline to make sense.
                    enable_remote_services=self.pipeline_options.enable_remote_services,
                    vlm_options=cast(ApiVlmOptions, self.pipeline_options.vlm_options),
                ),
            ]
        elif isinstance(self.pipeline_options.vlm_options, HuggingFaceVlmOptions):
            vlm_options = cast(HuggingFaceVlmOptions, self.pipeline_options.vlm_options)
            if vlm_options.inference_framework == InferenceFramework.MLX:
                self.build_pipe = [
                    HuggingFaceMlxModel(
                        enabled=True,  # must be always enabled for this pipeline to make sense.
                        artifacts_path=artifacts_path,
                        accelerator_options=pipeline_options.accelerator_options,
                        vlm_options=vlm_options,
                    ),
                ]
            else:
                self.build_pipe = [
                    HuggingFaceVlmModel(
                        enabled=True,  # must be always enabled for this pipeline to make sense.
                        artifacts_path=artifacts_path,
                        accelerator_options=pipeline_options.accelerator_options,
                        vlm_options=vlm_options,
                    ),
                ]

        self.enrichment_pipe = [
            # Other models working on `NodeItem` elements in the DoclingDocument
        ]

    def initialize_page(self, conv_res: ConversionResult, page: Page) -> Page:
        with TimeRecorder(conv_res, "page_init"):
            page._backend = conv_res.input._backend.load_page(page.page_no)  # type: ignore
            if page._backend is not None and page._backend.is_valid():
                page.size = page._backend.get_size()

        return page

    def extract_text_from_backend(
        self, page: Page, bbox: Union[BoundingBox, None]
    ) -> str:
        # Convert bounding box normalized to 0-100 into page coordinates for cropping
        text = ""
        if bbox:
            if page.size:
                if page._backend:
                    text = page._backend.get_text_in_rect(bbox)
        return text

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):

            if (
                self.pipeline_options.vlm_options.response_format
                == ResponseFormat.DOCTAGS
            ):
                doctags_list = []
                image_list = []
                for page in conv_res.pages:
                    predicted_doctags = ""
                    img = PILImage.new("RGB", (1, 1), "rgb(255,255,255)")
                    if page.predictions.vlm_response:
                        predicted_doctags = page.predictions.vlm_response.text
                    if page.image:
                        img = page.image
                    image_list.append(img)
                    doctags_list.append(predicted_doctags)

                doctags_list_c = cast(List[Union[Path, str]], doctags_list)
                image_list_c = cast(List[Union[Path, PILImage.Image]], image_list)
                doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
                    doctags_list_c, image_list_c
                )
                conv_res.document.load_from_doctags(doctags_doc)

                # If forced backend text, replace model predicted text with backend one
                if page.size:
                    if self.force_backend_text:
                        scale = self.pipeline_options.images_scale
                        for element, _level in conv_res.document.iterate_items():
                            if (
                                not isinstance(element, TextItem)
                                or len(element.prov) == 0
                            ):
                                continue
                            crop_bbox = (
                                element.prov[0]
                                .bbox.scaled(scale=scale)
                                .to_top_left_origin(
                                    page_height=page.size.height * scale
                                )
                            )
                            txt = self.extract_text_from_backend(page, crop_bbox)
                            element.text = txt
                            element.orig = txt
            elif (
                self.pipeline_options.vlm_options.response_format
                == ResponseFormat.MARKDOWN
            ):
                conv_res.document = self._turn_md_into_doc(conv_res)

            else:
                raise RuntimeError(
                    f"Unsupported VLM response format {self.pipeline_options.vlm_options.response_format}"
                )

            # Generate images of the requested element types
            if self.pipeline_options.generate_picture_images:
                scale = self.pipeline_options.images_scale
                for element, _level in conv_res.document.iterate_items():
                    if not isinstance(element, DocItem) or len(element.prov) == 0:
                        continue
                    if (
                        isinstance(element, PictureItem)
                        and self.pipeline_options.generate_picture_images
                    ):
                        page_ix = element.prov[0].page_no - 1
                        page = conv_res.pages[page_ix]
                        assert page.size is not None
                        assert page.image is not None

                        crop_bbox = (
                            element.prov[0]
                            .bbox.scaled(scale=scale)
                            .to_top_left_origin(page_height=page.size.height * scale)
                        )

                        cropped_im = page.image.crop(crop_bbox.as_tuple())
                        element.image = ImageRef.from_pil(
                            cropped_im, dpi=int(72 * scale)
                        )

            # 메타데이터 추출 (마지막에 추가)
            if self.pipeline_options.data_enrichment and conv_res.document:
                temp_content = ""
                total_pages = len(conv_res.document.pages)
                for page in range(1, min(3, total_pages+1)):
                    temp_content += conv_res.document.export_to_markdown(page_no=page)
                metadata = self.extract_document_metadata(temp_content)
                if metadata:
                    _log.info(f"추출된 메타데이터: {json.dumps(metadata, ensure_ascii=False, indent=2)}")

        return conv_res

    def _turn_md_into_doc(self, conv_res):
        predicted_text = ""
        for pg_idx, page in enumerate(conv_res.pages):
            if page.predictions.vlm_response:
                predicted_text += page.predictions.vlm_response.text + "\n\n"
        response_bytes = BytesIO(predicted_text.encode("utf8"))
        out_doc = InputDocument(
            path_or_stream=response_bytes,
            filename=conv_res.input.file.name,
            format=InputFormat.MD,
            backend=MarkdownDocumentBackend,
        )
        backend = MarkdownDocumentBackend(
            in_doc=out_doc,
            path_or_stream=response_bytes,
        )
        return backend.convert()

    @classmethod
    def get_default_options(cls) -> VlmPipelineOptions:
        return VlmPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return isinstance(backend, PdfDocumentBackend)

    def extract_document_metadata(self, document_content, model="google/gemma-3-12b-it:free", seed=3):
        """
        문서 내용에서 작성일과 작성자 정보를 추출하는 함수
        
        Args:
            document_content (str): 문서 내용
            model (str): 사용할 모델 이름
            seed (int): 재현성을 위한 시드값
            
        Returns:
            dict: 추출된 메타데이터 딕셔너리 (작성일, 작성자 정보)
        """
        if not self.pipeline_options.data_enrichment:
            return None
        
        # API 키 직접 설정
        api_key = "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"
        
        # OpenAI 클라이언트 초기화
        client = OpenAI(base_url="https://openrouter.ai/api/v1", 
                        api_key=api_key)
        
        # 프롬프트 구성
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a professional document extraction assistant. "
                            "Your job is to carefully extract metadata from semi-structured or unstructured Korean financial documents. "
                            "Always follow the requested output format exactly."
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
                            "다음은 한국어로 작성된 금융 보고서 문서의 일부입니다. 이 문서에서 아래 정보를 정확히 추출해주세요:\n\n"
                            "1. 최초 작성일 (날짜가 여러 개 있으면 '보고자료' 또는 '회의자료'에 가까운 날짜 사용)\n"
                            "2. 작성자 이름과 소속 또는 직책 (예: 채권시장팀 조용범 과장)\n"
                            "3. 작성자 전화번호 (내선번호 포함 시 함께 출력)\n\n"
                            "문서:\n"
                            "---\n"
                            f"{document_content}\n"
                            "---\n\n"
                            "출력 형식은 반드시 아래와 같이 맞춰주세요 (JSON 형식):\n\n"
                            "{\n"
                            '  "작성일": "YYYY-MM-DD",\n'
                            '  "작성자": [\n'
                            '    {"이름": "조용범", "소속": "채권시장팀", "직책": "과장", "전화": "4751"},\n'
                            '    {"이름": "임인혁", "소속": "주식시장팀", "직책": "과장", "전화": "4672"}\n'
                            "  ]\n"
                            "}"
                        )
                    }
                ]
            }
        ]
        
        try:
            # API 요청
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                seed=seed
            )
            
            # 응답에서 JSON 추출
            response = completion.choices[0].message.content
            
            # JSON 찾기
            json_pattern = r'```json\s*([\s\S]*?)\s*```'
            match = re.search(json_pattern, response)
            
            if match:
                try:
                    metadata = json.loads(match.group(1))
                    return metadata
                except:
                    return {"작성일": None, "작성자": []}
            else:
                try:
                    # JSON 블록이 없는 경우 전체 응답을 JSON으로 파싱 시도
                    return json.loads(response)
                except:
                    return {"작성일": None, "작성자": []}
        except Exception as e:
            _log.error(f"메타데이터 추출 중 오류 발생: {str(e)}")
            return {"작성일": None, "작성자": []}
