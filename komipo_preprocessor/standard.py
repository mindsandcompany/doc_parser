from __future__ import annotations

import json
import os
from pathlib import Path

# import fitz
from collections import defaultdict
from datetime import datetime
from typing import Optional, Iterable, Any, List, Dict, Tuple

# import fitz
from fastapi import Request
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_core.documents import Document
from pydantic import BaseModel

#from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
#from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
#from docling.datamodel.document import ConversionStatus
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    #EasyOcrOptions,
    #OcrEngine,
    #PdfBackend,
    PdfPipelineOptions,
    TableFormerMode,
)
# docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import ConversionResult
from docling_core.transforms.chunker import HierarchicalChunker as OrgHierarchicalChunker
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker as OrgHybridChunker
from docling_core.transforms.chunker import (
    BaseChunk,
    BaseChunker,
    DocChunk,
    DocMeta,
)
from docling_core.types import DoclingDocument

from docling_core.types.doc import (
    BoundingBox,
    #CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    #ImageRef,
    #ProvenanceItem,
    #Size,
    #TableCell,
    #TableData,
    #GroupItem,

    DocItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    PageItem
)
from collections import Counter
import re

# ============================================

import warnings
from typing import Iterable, Iterator, Optional, Union

from pydantic import BaseModel, ConfigDict, PositiveInt, TypeAdapter, model_validator
from typing_extensions import Self

try:
    import semchunk
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError:
    raise RuntimeError(
        "Module requires 'chunking' extra; to install, run: "
        "`pip install 'docling-core[chunking]'`"
    )

from genos_utils import upload_files

os.environ['HF_HOME'] = "/home/mnc/temp/.cache/huggingface"
####################################################
#################### 전처리 코드 #####################
####################################################
def get_max_page(result):
    max_page = 0
    for item, level in result.iterate_items():
        if isinstance(item, TextItem):
            if item.prov[0].page_no > max_page:
                max_page = item.prov[0].page_no
        elif isinstance(item, TableItem):
            if item.prov[0].page_no > max_page:
                max_page = item.prov[0].page_no
    return max_page

def get_max_height(result):
    max_height = 0
    page_header_texts = []
    page_heights = {}
    for item, level in result.iterate_items():
        if isinstance(item, TextItem):
            if item.label == 'page_header':
                page_header_texts.append(item.text)
        if item.prov[0].bbox.t > max_height:
            max_height = item.prov[0].bbox.t
        page_no = item.prov[0].page_no
        if page_no not in page_heights:
            page_heights[page_no] = int(item.prov[0].bbox.t)
        else:
            page_heights[page_no] = max(page_heights[page_no], int(item.prov[0].bbox.t))
    header_counter = Counter(page_header_texts)
    if len(header_counter.most_common()) > 0:
        page_header_text = header_counter.most_common(1)[0][0]
    else:
        page_header_text = ''
    return max_height, page_header_text, page_heights

# 테이블기준으로 삭제선 설정.
def get_delete_line_from_table(max_height, result, page_heights):
    max_below_line = max_height*0.8
    re_pattern_texts = r'^(절\s?차\s?서|(운\s?전\s?)?지\s?침\s?서|품\s?질\s?매\s?뉴\s?얼)$'
    re_pattern = r'^(페\s?이\s?지\s?:?|시\s?행\s?일\s?자\s?:?|절차서 번호|개\s?정\s?번\s?호\s?:?).*'
    re_pattern_2 = r'.*(페이지(\s?:\s?|\s)\d{1,3}\s?/\s?\d{1,3})$'
    delete_lines = []
    page_delete_lines = {}
    first_header_detect = {}
    def update_target_page(item):
        page_no = item.prov[0].page_no
        if page_no not in page_delete_lines:
            page_delete_lines[page_no] = int(item.prov[0].bbox.b)
        else:
            page_delete_lines[page_no] = min(page_delete_lines[page_no], int(item.prov[0].bbox.b))
    for item, level in result.iterate_items():
        if isinstance(item, TableItem):
            page_no = item.prov[0].page_no
            # 해당 페이지의 첫번째 헤더테이블을 찾았으면, 중단
            if page_no in first_header_detect and first_header_detect[page_no]:
                continue
            cnt = 0
            if item.prov[0].bbox.t > page_heights[page_no]*0.80 and item.prov[0].bbox.b > page_heights[page_no]*0.5:
                for cell in item.data.table_cells:
                    # 테이블내 셀값은 10개 이내로 조사. 엉뚱한 테이블이 걸리지 않기 위함.
                    cnt += 1
                    if cnt == 10:
                        break
                    if re.match(re_pattern_texts, cell.text):
                        update_target_page(item)
                        first_header_detect[page_no] = True
                    elif re.match(re_pattern, cell.text):
                        if cell.column_header == True and cell.text == "페이지":
                            continue
                        if item.prov[0].bbox.b < max_below_line:
                            max_below_line = item.prov[0].bbox.b
                        update_target_page(item)
                        delete_lines.append(int(item.prov[0].bbox.b))
                        first_header_detect[page_no] = True
                    elif re.match(re_pattern_2, cell.text):
                        update_target_page(item)
                        first_header_detect[page_no] = True
    # 빈도수가 가장 높은 삭제선을 대표값으로 설정.
    delete_line_counter = Counter(delete_lines)
    if len(delete_line_counter.most_common()) > 0:
        delete_line = delete_line_counter.most_common(1)[0][0]
    else:
        delete_line = None
    return delete_line, page_delete_lines

# 테이블이 없을경우, 
def update_delete_line_from_text(header_table_delete_line, result, page_delete_lines, max_height):
    re_pattern_texts = r'^(절\s?차\s?서|(운\s?전\s?)?지\s?침\s?서|품\s?질\s?매\s?뉴\s?얼)$'
    re_pattern = r"(^((페|폐)\s?이\s?지\s?:?|시\s?행\s?일\s?자\s?:?|절차서 번호).*|.*시행일자\s?:\s?('\d{2}.\d{2}|\d{4}.\s?\d{2}.?)$)"
    re_pattern_2 = r'.*(페이지(\s?:\s?|\s)\d{1,3}\s?/\s?\d{1,3})$'
    re_pattern_page_no = r'^\d{1,3}\s?의\s?\d{1,3}$'
    re_pattern_3 = r'.*(페\s?이\s?지\s?:\s?(\d{1,3})?)$'
    re_pattern_3_page_no = r'^\d{1,3}\s?/\s?\d{1,3}$'
    if header_table_delete_line:
        updated_delete_line = header_table_delete_line
    else:
        updated_delete_line = max_height*0.8
    page_num_trigger = {} # 헤더부분 "페이지" 텍스트를 찾았을경우, 아래 2번째 라인의 날짜 형식을 찾기위한 변수.
    page_delete_line_update = {} # 양면 문서 우측의 첫번째 텍스트를 찾았을경우, 더이상 업데이트 하지 않기 위한 변수.
    initial_page_delete_lines = page_delete_lines.copy()
    page_delete_lines_text = {} # 현재 함수에서 얻어진 삭제선.
    def update_target_page(item, item_above_flag):
        page_no = item.prov[0].page_no
        if item_above_flag:
            if page_no in page_delete_line_update and page_delete_line_update[page_no] == True:
                pass
            else:
                if page_no in page_delete_lines:
                    del page_delete_lines[page_no]
                    page_delete_line_update[page_no] = True
        if page_no not in page_delete_lines:
            page_delete_lines[page_no] = int(item.prov[0].bbox.b)
        else:
            page_delete_lines[page_no] = min(page_delete_lines[page_no], int(item.prov[0].bbox.b))
        
        if page_no not in page_delete_lines_text:
            page_delete_lines_text[page_no] = int(item.prov[0].bbox.b)
        else:
            page_delete_lines_text[page_no] = min(page_delete_lines_text[page_no], int(item.prov[0].bbox.b))

    for item, level in result.iterate_items():
        item_above_flag = False # 양면 문서에서 좌측에 테이블이 있어서, 우측의 본문이 잘린 경우를 위한 변수.
        if isinstance(item, TextItem):
            page_no = item.prov[0].page_no
            if page_no in initial_page_delete_lines:
                # 테이블로 정해진 삭제선이 있고, 그보다 작으면 조사를 안해도 되지만, 
                if item.prov[0].bbox.t < initial_page_delete_lines[page_no]:
                    continue
                else:
                    # 삭제선보다 높은위치에 있는 텍스트는 다시 조사.
                    # 양면 문서의 우측에 텍스트가 존재하는 문서를 대비함.
                    item_above_flag = True
            if item.prov[0].bbox.t > updated_delete_line:
                if page_no in page_num_trigger and page_num_trigger[page_no]:
                    if re.match(re_pattern_page_no, item.text) or re.match(re_pattern_3_page_no, item.text):
                        update_target_page(item, item_above_flag)
                if re.match(re_pattern_texts, item.text):
                    update_target_page(item, item_above_flag)
                elif re.match(re_pattern, item.text):
                    update_target_page(item, item_above_flag)
                elif re.match(re_pattern_2, item.text):
                    update_target_page(item, item_above_flag)
                    page_num_trigger[page_no] = True
                elif re.match(re_pattern_3, item.text):
                    update_target_page(item, item_above_flag)
                    page_num_trigger[page_no] = True
            elif item.prov[0].bbox.t > updated_delete_line * 0.6:
                if page_no in page_num_trigger and page_num_trigger[page_no]:
                    if re.match(re_pattern_page_no, item.text) or re.match(re_pattern_3_page_no, item.text):
                        update_target_page(item, item_above_flag)
                if re.match(re_pattern, item.text) or re.match(re_pattern_2, item.text):
                    update_target_page(item, item_above_flag)
                    if item.prov[0].bbox.b < updated_delete_line:
                        updated_delete_line = item.prov[0].bbox.b
                    page_num_trigger[page_no] = True
                elif re.match(re_pattern_3, item.text):
                    update_target_page(item, item_above_flag)
                    if item.prov[0].bbox.b < updated_delete_line:
                        updated_delete_line = item.prov[0].bbox.b
                    page_num_trigger[page_no] = True
            if re.match(re_pattern, item.text) or re.match(re_pattern_2, item.text) or re.match(re_pattern_3, item.text):
                if page_no in page_num_trigger and page_num_trigger[page_no]:
                    if re.match(re_pattern_page_no, item.text) or re.match(re_pattern_3_page_no, item.text):
                        update_target_page(item, item_above_flag)
                if item.prov[0].bbox.t < max_height * 0.5:
                    continue
                elif re.match(r'^(절차서 제목)', item.text):
                    continue
                update_target_page(item, item_above_flag)
                page_num_trigger[page_no] = True

    for item, level in result.iterate_items():
        item_above_flag = False # 양면 문서에서 좌측에 테이블이 있어서, 우측의 본문이 잘린 경우를 위한 변수.
        if isinstance(item, TextItem):
            page_no = item.prov[0].page_no
            if page_no in page_delete_lines_text and item.prov[0].bbox.t > page_delete_lines_text[page_no] and item.prov[0].bbox.b < page_delete_lines_text[page_no]:
                if page_no in page_delete_lines:
                    if page_delete_lines[page_no] > item.prov[0].bbox.b:
                        page_delete_lines[page_no] = item.prov[0].bbox.b
    return page_delete_lines

def process_pdf(org_document: DoclingDocument):

    origin = DocumentOrigin(
        filename=org_document.origin.filename,
        mimetype="application/pdf",
        binary_hash=org_document.origin.binary_hash,
    )

    new_doc = DoclingDocument(
        name="file", origin=origin
    )
    max_height, page_header_text, page_heights = get_max_height(org_document)
    header_table_delete_line, page_delete_lines = get_delete_line_from_table(max_height, org_document, page_heights)
    updated_page_delete_lines = update_delete_line_from_text(header_table_delete_line, org_document, page_delete_lines, max_height)

    section_header_cache = []
    last_level = 0
    def add_headers(header, new_doc, last_level):
        last_level += 1
        new_doc.add_heading(
            text=header.text,
            orig=header.text,
            level=last_level,
            prov=header.prov[0],
            parent=new_doc.body
        )
        return last_level

    def attach_text(item, new_doc):
        nonlocal section_header_cache
        nonlocal last_level
        if item.label in ['page_header', 'caption']:
            new_doc.add_heading(
                text=item.text,
                orig=item.text,
                prov=item.prov[0],
                parent=new_doc.body
            )
        elif item.label == 'section_header':
            section_header_cache.append(item)
        elif item.label=='list_item':
            if len(section_header_cache)>0:
                for header in section_header_cache:
                    last_level = add_headers(header, new_doc, last_level)
                section_header_cache = []
                last_level = 0
            new_doc.add_list_item(
                text=item.text,
                enumerated=item.enumerated,
                marker=item.marker,
                orig=item.text,
                prov=item.prov[0],
                parent=new_doc.body
            )
        elif item.label=='text' or item.label=='checkbox_unselected' or item.label=='code' or item.label=='paragraph':
            # print("text", item.self_ref)
            if len(section_header_cache)>0:
                for header in section_header_cache:
                    last_level = add_headers(header, new_doc, last_level)
                section_header_cache = []
                last_level = 0
            new_doc.add_text(
                label=item.label,
                text=item.text,
                orig=item.text,
                prov=item.prov[0],
                parent=new_doc.body
            )

    for key, item in org_document.pages.items():
        if isinstance(item, PageItem):
            new_doc.add_page(
                page_no=item.page_no,
                size=item.size,
                image=item.image,
            )
    label_list = ['page_header', 'section_header', 'list_item']
    for item, level in org_document.iterate_items():
        page_no = item.prov[0].page_no
        if item.prov[0].page_no != 1:
            if isinstance(item, TextItem) and ''.join(item.text.split()) == ''.join(page_header_text.split()):
                if item.label in label_list:
                    continue
                elif page_no in updated_page_delete_lines and item.prov[0].bbox.t > updated_page_delete_lines[page_no]:
                    continue
            if page_no in updated_page_delete_lines and item.prov[0].bbox.t > updated_page_delete_lines[page_no]:
                if item.prov[0].bbox.b < updated_page_delete_lines[page_no]:
                    middle_line = item.prov[0].bbox.b + (item.prov[0].bbox.t - item.prov[0].bbox.b)/2
                    if middle_line > updated_page_delete_lines[page_no]:
                        continue
                    else:
                        pass
                else:
                    continue
        if isinstance(item, PictureItem):
            if len(section_header_cache)>0:
                for header in section_header_cache:
                    last_level = add_headers(header, new_doc, last_level)
                section_header_cache = []
                last_level = 0
            new_doc.add_picture(
                annotations=[],
                image=item.image,
                caption=[],
                prov=item.prov[0],
                parent=new_doc.body
            )
            if len(item.children) > 0:
                for child in item.children:
                    for text in org_document.texts:
                        if text.self_ref == child.cref:
                            attach_text(text, new_doc)
                            break
        elif isinstance(item, TableItem):
            if len(section_header_cache)>0:
                for header in section_header_cache:
                    last_level = add_headers(header, new_doc, last_level)
                section_header_cache = []
                last_level = 0
            new_doc.add_table(
                data=item.data,
                caption=[],
                prov=item.prov[0],
                parent=new_doc.body
            )
        elif isinstance(item, TextItem):
            attach_text(item, new_doc)

    return new_doc

####################################################
#################### 전처리 코드 #####################
####################################################

#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Chunker implementation leveraging the document structure."""

from typing import Any, ClassVar, Final, Iterator, Literal, Optional

from pandas import DataFrame
from pydantic import Field, StringConstraints, field_validator

#from docling_core.transforms.chunker import BaseChunk, BaseChunker, BaseMeta
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.document import (
    #DocItem,
    DocumentOrigin,
    LevelNumber,
    ListItem,
    #SectionHeaderItem,
    #TableItem,
    #TextItem,
)
from docling_core.types.doc.labels import DocItemLabel


#class HierarchicalChunker(OrgHierarchicalChunker):
    
class HierarchicalChunker(BaseChunker):
    r"""Chunker implementation leveraging the document layout.

    Args:
        merge_list_items (bool): Whether to merge successive list items.
            Defaults to True.
        delim (str): Delimiter to use for merging text. Defaults to "\n".
    """

    merge_list_items: bool = True
    delim: str = "\n"

    @classmethod
    def _triplet_serialize(cls, table_df: DataFrame) -> str:

        # copy header as first row and shift all rows by one
        table_df.loc[-1] = table_df.columns  # type: ignore[call-overload]
        table_df.index = table_df.index + 1
        table_df = table_df.sort_index()

        rows = [str(item).strip() for item in table_df.iloc[:, 0].to_list()]
        cols = [str(item).strip() for item in table_df.iloc[0, :].to_list()]

        nrows = table_df.shape[0]
        ncols = table_df.shape[1]
        texts = [
            f"{rows[i]}, {cols[j]} = {str(table_df.iloc[i, j]).strip()}"
            for i in range(1, nrows)
            for j in range(1, ncols)
        ]
        output_text = ". ".join(texts)

        return output_text

    def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        r"""Chunk the provided document.

        Args:
            dl_doc (DLDocument): document to chunk

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        heading_by_level: dict[LevelNumber, str] = {}
        list_items: list[TextItem] = []
        for item, level in dl_doc.iterate_items():
            captions = None
            if isinstance(item, DocItem):

                # first handle any merging needed
                if self.merge_list_items:
                    if isinstance(
                        item, ListItem
                    ) or (  # TODO remove when all captured as ListItem:
                        isinstance(item, TextItem)
                        and item.label == DocItemLabel.LIST_ITEM
                    ):
                        list_items.append(item)
                        continue
                    elif list_items:  # need to yield
                        yield DocChunk(
                            text=self.delim.join([i.text for i in list_items]),
                            meta=DocMeta(
                                doc_items=list_items,
                                headings=[
                                    heading_by_level[k]
                                    for k in sorted(heading_by_level) if k != 4
                                ]
                                or None,
                                origin=dl_doc.origin,
                            ),
                        )
                        list_items = []  # reset

                if isinstance(item, SectionHeaderItem) or (
                    isinstance(item, TextItem)
                    and item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]
                ):
                    level = (
                        item.level
                        if isinstance(item, SectionHeaderItem)
                        else (0 if item.label == DocItemLabel.TITLE else 1)
                    )
                    heading_by_level[level] = item.text

                    # remove headings of higher level as they just went out of scope
                    keys_to_del = [k for k in heading_by_level if k > level]
                    for k in keys_to_del:
                        heading_by_level.pop(k, None)
                    continue

                if isinstance(item, TextItem) or (
                    (not self.merge_list_items) and isinstance(item, ListItem)
                ):
                    text = item.text
                elif isinstance(item, TableItem):
                    table_df = item.export_to_dataframe()
                    if table_df.shape[0] < 1 or table_df.shape[1] < 1:
                        # at least two cols needed, as first column contains row headers
                        continue
                    text = self._triplet_serialize(table_df=table_df)
                    captions = [
                        c.text for c in [r.resolve(dl_doc) for r in item.captions]
                    ] or None
                elif isinstance(item, PictureItem):
                    text = ''.join(str(value) for value in heading_by_level.values())
                else:
                    continue
                c = DocChunk(
                    text=text,
                    meta=DocMeta(
                        doc_items=[item],
                        headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                        or None,
                        captions=captions,
                        origin=dl_doc.origin,
                    ),
                )
                yield c

        if self.merge_list_items and list_items:  # need to yield
            yield DocChunk(
                text=self.delim.join([i.text for i in list_items]),
                meta=DocMeta(
                    doc_items=list_items,
                    headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                    or None,
                    origin=dl_doc.origin,
                ),
            )

class HybridChunker(BaseChunker):
    r"""Chunker doing tokenization-aware refinements on top of document layout chunking.

    Args:
        tokenizer: The tokenizer to use; either instantiated object or name or path of
            respective pretrained model
        max_tokens: The maximum number of tokens per chunk. If not set, limit is
            resolved from the tokenizer
        merge_peers: Whether to merge undersized chunks sharing same relevant metadata
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Union[PreTrainedTokenizerBase, str] = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    max_tokens: int = None  # type: ignore[assignment]
    merge_peers: bool = True

    _inner_chunker: HierarchicalChunker = HierarchicalChunker()

    @model_validator(mode="after")
    def _patch_tokenizer_and_max_tokens(self) -> Self:
        self._tokenizer = (
            self.tokenizer
            if isinstance(self.tokenizer, PreTrainedTokenizerBase)
            else AutoTokenizer.from_pretrained(self.tokenizer)
        )
        if self.max_tokens is None:
            self.max_tokens = TypeAdapter(PositiveInt).validate_python(
                self._tokenizer.model_max_length
            )
        return self

    def _count_tokens(self, text: Optional[Union[str, list[str]]]):
        if text is None:
            return 0
        elif isinstance(text, list):
            total = 0
            for t in text:
                total += self._count_tokens(t)
            return total
        return len(self._tokenizer.tokenize(text, max_length=None))

    class _ChunkLengthInfo(BaseModel):
        total_len: int
        text_len: int
        other_len: int

    def _doc_chunk_length(self, doc_chunk: DocChunk):
        text_length = self._count_tokens(doc_chunk.text)
        headings_length = self._count_tokens(doc_chunk.meta.headings)
        captions_length = self._count_tokens(doc_chunk.meta.captions)
        total = text_length + headings_length + captions_length
        return self._ChunkLengthInfo(
            total_len=total,
            text_len=text_length,
            other_len=total - text_length,
        )

    def _make_chunk_from_doc_items(
        self, doc_chunk: DocChunk, window_text: str, window_start: int, window_end: int
    ):
        meta = DocMeta(
            doc_items=doc_chunk.meta.doc_items[window_start : window_end + 1],
            headings=doc_chunk.meta.headings,
            captions=doc_chunk.meta.captions,
            origin=doc_chunk.meta.origin,
        )
        new_chunk = DocChunk(text=window_text, meta=meta)
        return new_chunk

    def _merge_text(self, t1, t2):
        if t1 == "":
            return t2
        elif t2 == "":
            return t1
        else:
            return f"{t1}{self.delim}{t2}"

    def _split_by_doc_items(self, doc_chunk: DocChunk) -> list[DocChunk]:
        if doc_chunk.meta.doc_items is None or len(doc_chunk.meta.doc_items) <= 1:
            return [doc_chunk]
        length = self._doc_chunk_length(doc_chunk)
        if length.total_len <= self.max_tokens:
            return [doc_chunk]
        else:
            chunks = []
            window_start = 0
            window_end = 0
            window_text = ""
            window_text_length = 0
            other_length = length.other_len
            num_items = len(doc_chunk.meta.doc_items)
            while window_end < num_items:
                doc_item = doc_chunk.meta.doc_items[window_end]
                if isinstance(doc_item, TextItem):
                    text = doc_item.text
                else:
                    raise RuntimeError("Non-TextItem split not implemented yet")
                text_length = self._count_tokens(text)
                if (
                    text_length + window_text_length + other_length < self.max_tokens
                    and window_end < num_items - 1
                ):
                    # Still room left to add more to this chunk AND still at least one
                    # item left
                    window_end += 1
                    window_text_length += text_length
                    window_text = self._merge_text(window_text, text)
                elif text_length + window_text_length + other_length < self.max_tokens:
                    # All the items in the window fit into the chunk and there are no
                    # other items left
                    window_text = self._merge_text(window_text, text)
                    new_chunk = self._make_chunk_from_doc_items(
                        doc_chunk, window_text, window_start, window_end
                    )
                    chunks.append(new_chunk)
                    window_end = num_items
                elif window_start == window_end:
                    # Only one item in the window and it doesn't fit into the chunk. So
                    # we'll just make it a chunk for now and it will get split in the
                    # plain text splitter.
                    window_text = self._merge_text(window_text, text)
                    new_chunk = self._make_chunk_from_doc_items(
                        doc_chunk, window_text, window_start, window_end
                    )
                    chunks.append(new_chunk)
                    window_start = window_end + 1
                    window_end = window_start
                    window_text = ""
                    window_text_length = 0
                else:
                    # Multiple items in the window but they don't fit into the chunk.
                    # However, the existing items must have fit or we wouldn't have
                    # gotten here. So we put everything but the last item into the chunk
                    # and then start a new window INCLUDING the current window end.
                    new_chunk = self._make_chunk_from_doc_items(
                        doc_chunk, window_text, window_start, window_end - 1
                    )
                    chunks.append(new_chunk)
                    window_start = window_end
                    window_text = ""
                    window_text_length = 0
            return chunks

    def _split_using_plain_text(
        self,
        doc_chunk: DocChunk,
    ) -> list[DocChunk]:
        lengths = self._doc_chunk_length(doc_chunk)
        if lengths.total_len <= self.max_tokens:
            # data = [DocChunk(**doc_chunk.export_json_dict())]
            data = [DocChunk(text=doc_chunk.text, meta=doc_chunk.meta)]
            return data
        else:
            # How much room is there for text after subtracting out the headers and
            # captions:
            available_length = self.max_tokens - lengths.other_len
            sem_chunker = semchunk.chunkerify(
                self._tokenizer, chunk_size=available_length
            )
            if available_length <= 0:
                warnings.warn(
                    f"Headers and captions for this chunk are longer than the total amount of size for the chunk, chunk will be ignored: {doc_chunk.text=}"  # noqa
                )
                return []
            text = doc_chunk.text
            segments = sem_chunker.chunk(text)
            chunks = [DocChunk(text=s, meta=doc_chunk.meta) for s in segments]
            return chunks

    def _merge_chunks_with_matching_metadata(self, chunks: list[DocChunk]):
        output_chunks = []
        window_start = 0
        window_end = 0
        num_chunks = len(chunks)
        while window_end < num_chunks:
            chunk = chunks[window_end]
            lengths = self._doc_chunk_length(chunk)
            headings_and_captions = (chunk.meta.headings, chunk.meta.captions)
            ready_to_append = False
            if window_start == window_end:
                # starting a new block of chunks to potentially merge
                current_headings_and_captions = headings_and_captions
                window_text = chunk.text
                window_other_length = lengths.other_len
                window_text_length = lengths.text_len
                window_items = chunk.meta.doc_items
                window_end += 1
                first_chunk_of_window = chunk
            elif (
                headings_and_captions == current_headings_and_captions
                and window_text_length + window_other_length + lengths.text_len
                <= self.max_tokens
            ):
                # there is room to include the new chunk so add it to the window and
                # continue
                window_text = self._merge_text(window_text, chunk.text)
                window_text_length += lengths.text_len
                window_items = window_items + chunk.meta.doc_items
                window_end += 1
            else:
                ready_to_append = True

            if ready_to_append or window_end == num_chunks:
                # no more room OR the start of new metadata.  Either way, end the block
                # and use the current window_end as the start of a new block
                if window_start + 1 == window_end:
                    # just one chunk so use it as is
                    output_chunks.append(first_chunk_of_window)
                else:
                    new_meta = DocMeta(
                        doc_items=window_items,
                        headings=current_headings_and_captions[0],
                        captions=current_headings_and_captions[1],
                        origin=chunk.meta.origin,
                    )
                    new_chunk = DocChunk(
                        text=window_text,
                        meta=new_meta,
                    )
                    output_chunks.append(new_chunk)
                # no need to reset window_text, etc. because that will be reset in the
                # next iteration in the if window_start == window_end block
                window_start = window_end

        return output_chunks

    def chunk(self, dl_doc: DoclingDocument, **kwargs) -> Iterator[BaseChunk]:
        r"""Chunk the provided document.

        Args:
            dl_doc (DLDocument): document to chunk

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        res: Iterable[DocChunk]
        res = self._inner_chunker.chunk(dl_doc=dl_doc, **kwargs)  # type: ignore
        res = [x for c in res for x in self._split_by_doc_items(c)]
        res = [x for c in res for x in self._split_using_plain_text(c)]
        if self.merge_peers:
            res = self._merge_chunks_with_matching_metadata(res)
        return iter(res)


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
    page_bboxes: list = None

class GenOSVectorMetaBuilder:
    def __init__(self):
        """빌더 초기화"""
        self.text: Optional[str] = None
        self.n_char: Optional[int] = 0
        self.n_word: Optional[int] = 0
        self.n_line: Optional[int] = 0
        self.i_page: Optional[int] = None
        self.i_chunk_on_page: Optional[int] = None
        self.n_chunk_of_page: Optional[int] = None
        self.i_chunk_on_doc: Optional[int] = None
        self.n_chunk_of_doc: Optional[int] = None
        self.n_page: Optional[int] = 0
        self.reg_date: Optional[str] = None
        self.bboxes: str = None
        self.chunk_bboxes: list = None
        self.media_files: list = None


    def set_text(self, text: str) -> "GenOSVectorMetaBuilder":
        """텍스트와 관련된 데이터를 설정"""
        self.text = text
        self.n_char = len(text)
        self.n_word = len(text.split())
        self.n_line = len(text.splitlines())
        return self

    def set_page_info(
            self, i_page: int, i_chunk_on_page: int, n_chunk_of_page: int
    ) -> "GenOSVectorMetaBuilder":
        """페이지 정보 설정"""
        self.i_page = i_page
        self.i_chunk_on_page = i_chunk_on_page
        self.n_chunk_of_page = n_chunk_of_page
        return self

    def set_chunk_index(self, i_chunk_on_doc: int) -> "GenOSVectorMetaBuilder":
        """문서 전체의 청크 인덱스 설정"""
        self.i_chunk_on_doc = i_chunk_on_doc
        return self

    def set_bboxes(self, bbox: BoundingBox) -> "GenOSVectorMetaBuilder":
        """Bounding Boxes 정보 설정"""
        #         bboxes.append({
        #             'p1': {'x': rect[0] / fitz_page.rect.width, 'y': rect[1] / fitz_page.rect.height},
        #             'p2': {'x': rect[2] / fitz_page.rect.width, 'y': rect[3] / fitz_page.rect.height},
        #         })
        # NOTE: docling은 BOTTOMLEFT인데 해당 좌표 그대로 활용되는지 ?
        conv = []
        conv.append({
            'p1': {'x': bbox.l, 'y': bbox.t},
            'p2': {'x': bbox.r, 'y': bbox.b},
        })
        self.bboxes = json.dumps(conv)
        return self

    def set_global_metadata(self, **global_metadata) -> "GenOSVectorMetaBuilder":
        """글로벌 메타데이터 병합"""
        for key, value in global_metadata.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def set_chunk_bboxes(self, doc_items: list, document: DoclingDocument) -> "GenOSVectorMetaBuilder":
        self.chunk_bboxes = []
        for item in doc_items:
            for prov in item.prov:
                label = item.self_ref
                type_ = item.label
                size = document.pages.get(prov.page_no).size
                page_no = prov.page_no
                bbox = prov.bbox
                bbox_data = {'l': bbox.l / size.width,
                't': bbox.t / size.height,
                'r': bbox.r / size.width,
                'b': bbox.b / size.height,
                'coord_origin': bbox.coord_origin.value}
                self.chunk_bboxes.append({'page': page_no, 'bbox': bbox_data, 'type': type_, 'ref': label})
        return self

    def set_media_files(self, doc_items: list) -> "GenOSVectorMetaBuilder":
        temp_list = []
        for item in doc_items:
            if isinstance(item, PictureItem):
                path = str(item.image.uri)
                name = path.rsplit("/",1)[-1]
                temp_list.append({'name': name, 'type': 'image' })
                # temp_list.append(name)
        self.media_files = temp_list
        return self

    def build(self) -> GenOSVectorMeta:
        """설정된 데이터를 사용해 최종적으로 GenOSVectorMeta 객체 생성"""
        return GenOSVectorMeta(
            text=self.text,
            n_char=self.n_char,
            n_word=self.n_word,
            n_line=self.n_line,
            i_page=self.i_page,
            i_chunk_on_page=self.i_chunk_on_page,
            n_chunk_of_page=self.n_chunk_of_page,
            i_chunk_on_doc=self.i_chunk_on_doc,
            n_chunk_of_doc=self.n_chunk_of_doc,
            n_page=self.n_page,
            reg_date=self.reg_date,
            bboxes=self.bboxes,
            chunk_bboxes=self.chunk_bboxes,
            media_files=self.media_files,
        )


class DocumentProcessor:

    def __init__(self):
        '''
        initialize Document Converter
        '''
        self.page_chunk_counts = defaultdict(int)
        device = AcceleratorDevice.AUTO
        num_threads = 4
        table_mode = TableFormerMode.FAST
        accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)
        pipe_line_options = PdfPipelineOptions()
        pipe_line_options.generate_page_images = True
        # pipe_line_options.generate_table_images = True
        pipe_line_options.generate_picture_images = True
        pipe_line_options.do_ocr = True
        pipe_line_options.ocr_options.lang = ["ko", 'en']
        pipe_line_options.ocr_options.model_storage_directory = "/nfs-root/aiModel/.EasyOCR/model"
        #pipe_line_options.ocr_options.model_storage_directory = "/home/mnc/temp/.EasyOCR/model"
        #pipe_line_options.artifacts_path = Path("/home/mnc/temp/.cache/huggingface/hub/models--ds4sd--docling-models/snapshots/36bebf56681740529abd09f5473a93a69373fbf0/")
        pipe_line_options.do_table_structure = True
        pipe_line_options.images_scale = 2
        pipe_line_options.table_structure_options.do_cell_matching = True
        pipe_line_options.accelerator_options = accelerator_options

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipe_line_options,
                    backend=DoclingParseV2DocumentBackend
                )
            }
        )

    def load_documents_with_docling(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        # docling 설정
        # 필요에 따라 사용자 지정 가능. 여기서는 genos_vanilla 와 비슷하게 PDF를 처리한다 가정.
        # TODO: kwargs 와의 연결
        # TODO: Langchain document 를 꼭 써야하나?
        # 실제 변환 실행
        # ConversionResult 리스트를 받는다.
        #
        # NOTE: 처리시 파일 하나를 병렬로 처리하는지?? 아니면 폴더 단위로 병렬 처리 하는지??
        # NOTE: 파일 하나 처리 시 convert로 변경.
        # conv_results = doc_converter.convert_all([file_path], raises_on_error=True)
        conv_result: ConversionResult = self.converter.convert(file_path, raises_on_error=False)
        return process_pdf(conv_result.document)

    def load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        # ducling 방식으로 문서 로드
        return self.load_documents_with_docling(file_path, **kwargs)
        # return documents

    def split_documents(self, documents: DoclingDocument, **kwargs: dict) -> List[DocChunk]:
        ##FIXME: Hierarchical Chunker 로 수정
        ## NOTE: HierarchicalChunker -> HybridChunker로 공식 문서 변경 됨.
        ## 관련 url : https://ds4sd.github.io/docling/usage/#chunking
        ## 관련 url : https://ds4sd.github.io/docling/examples/hybrid_chunking/
        chunker: HybridChunker = HybridChunker()
        # TODO: 전처리 코드 넣기
        
        # TODO: 최종 전처리 document로 chunk 추출
        # chunker: HierarchicalChunker = HierarchicalChunker()
        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=documents, **kwargs))
        # TODO: 페이지 관련 이슈 처리 시 해당 부분도 수정 필요
        for chunk in chunks:
            self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks

    def safe_join(self, iterable):
        if not isinstance(iterable, (list, tuple, set)):
            return ''
        return ''.join(map(str, iterable)) + '\n'
    
    def remove_extension(self, filename):
        if filename.endswith(".pdf"):
            filename = filename.rsplit(".pdf", 1)[0]
        if filename.endswith(".pdf") or filename.endswith(".hwp"):
            filename = filename.rsplit(".", 1)[0]
        return filename+", "

    async def compose_vectors(self, document: DoclingDocument, chunks: List[DocChunk], file_path: str, request: Request, **kwargs: dict) -> \
            list[dict]:
        pdf_path = file_path.replace('.hwp', '.pdf').replace('.txt', '.pdf').replace('.json', '.pdf')

        # if os.path.exists(pdf_path):
        #     doc = fitz.open(pdf_path)
        print(kwargs)
        filename = kwargs.get('FILENAME', "==filename is empty.==")
        
        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=document.num_pages(),
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z',
        )

        current_page = None
        chunk_index_on_page = 0
        vectors = []

        for chunk_idx, chunk in enumerate(chunks):
            ## NOTE: chunk가 두 페이지에 걸쳐 있는 경우 첫번째 아이템을 사용
            ## NOTE: chunk가 두 페이지에 걸쳐서 있는 경우 bounding box 처리를 어떻게 해야하는 지...
            ## NOTE: 현재 구조에서는 처리가 불가
            ## NOTE: 임시로 페이지 넘어가는 경우 chunk를 분할해서 처리
            chunk_page = chunk.meta.doc_items[0].prov[0].page_no
            content = self.remove_extension(filename) + self.safe_join(chunk.meta.headings) + chunk.text
            vector = (GenOSVectorMetaBuilder()
                      .set_text(content)
                      .set_page_info(chunk_page, chunk_index_on_page, self.page_chunk_counts[chunk_page])
                      .set_chunk_index(chunk_idx)
                      .set_bboxes(chunk.meta.doc_items[0].prov[0].bbox)
                      .set_global_metadata(**global_metadata)
                      .set_chunk_bboxes(chunk.meta.doc_items, document)
                      .set_media_files(chunk.meta.doc_items)
                      ).build()
            vectors.append(vector)

            # page = chunk_page
            # text = chunk.page_content

            if chunk_page != current_page:
                current_page = chunk_page
                chunk_index_on_page = 0

            # bboxes_json = None
            # if os.path.exists(pdf_path):
            #     fitz_page = doc.load_page(page)
            #     bboxes = []
            #     # text 검색 시 fitz의 search_for 문맥이 주어진 text chunk 에 매칭되는 바운딩박스를 찾을 수 있는지 확인
            #     # 많은 경우 chunk가 PDF 내 같은 text를 그대로 match하지 못할 수 있음.
            #     # 여기서는 원본 genos_vanilla와 동일한 로직 유지.
            #     # 특정 성능 문제나 결과 없을 경우 try-except 추가 가능.
            #     search_results = fitz_page.search_for(text)
            #     for rect in search_results:
            #         bboxes.append({
            #             'p1': {'x': rect[0] / fitz_page.rect.width, 'y': rect[1] / fitz_page.rect.height},
            #             'p2': {'x': rect[2] / fitz_page.rect.width, 'y': rect[3] / fitz_page.rect.height},
            #         })
            #     bboxes_json = json.dumps(bboxes)

            chunk_index_on_page += 1

            file_list = self.get_media_files(chunk.meta.doc_items)
            await upload_files(file_list, request=request)

        return vectors

    def get_media_files(self, doc_items: list):
        temp_list = []
        for item in doc_items:
            if isinstance(item, PictureItem):
                path = str(item.image.uri)
                name = path.rsplit("/",1)[-1]
                temp_list.append({ 'path': path, 'name': name})
                #temp_list.append(path)

        return temp_list

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        document: DoclingDocument = self.load_documents(file_path, **kwargs)
        await assert_cancelled(request)

        output_path, output_file = os.path.split(file_path)
        filename, _ = os.path.splitext(output_file)
        artifacts_dir = Path(f"{output_path}/{filename}")
        #artifacts_dir = os.path.join(output_path, filename)
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = artifacts_dir.parent
        document = document._with_pictures_refs(image_dir=artifacts_dir, reference_path=reference_path)

        # Extract Chunk from DoclingDocument
        chunks: List[DocChunk] = self.split_documents(document, **kwargs)
        await assert_cancelled(request)

        # vectors: list[dict] = self.compose_vectors(document, chunks, file_path, **kwargs)
        # print(chunks)


        vectors = []
        if len(chunks) > 1:
            vectors: list[dict] = await self.compose_vectors(document, chunks, file_path, request, **kwargs)
        else:
            raise GenosServiceException(1, f"chunk length is 0")
        
        '''
        # 미디어 파일 업로드 방법
        media_files = [
            { 'path': '/tmp/graph.jpg', 'name': 'graph.jpg', 'type': 'image' },
            { 'path': '/result/1/graph.jpg', 'name': '1/graph.jpg', 'type': 'image' },
        ]

        # 업로드 요청 시에는 path, name 필요
        file_list = [{k: v for k, v in file.items() if k != 'type'} for file in media_files]
        await upload_files(file_list, request=request)

        # 메타에 저장시에는 name, type 필요
        meta = [{k: v for k, v in file.items() if k != 'path'} for file in media_files]
        vectors[0].media_files = meta
        '''
        
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
