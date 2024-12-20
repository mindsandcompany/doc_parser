from __future__ import annotations

import json
import os

import fitz
from collections import defaultdict
from datetime import datetime
from typing import Optional, Iterable, Any, List, Dict, Tuple

# import fitz
from fastapi import Request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
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
from docling.datamodel.document import ConversionResult
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker import (
    BaseChunk,
    BaseChunker,
    DocChunk,
    DocMeta,
)
from docling_core.types import DoclingDocument

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    ImageRef,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
    GroupItem,

    DocItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    PageItem
)
from collections import Counter
import re


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

def get_delete_line_from_table(max_height, result, page_heights):
    max_below_line = max_height * 0.8
    target_page = set()
    re_pattern_texts = r'^절\s?차\s?서$|^품\s?질\s?매\s?뉴\s?얼$'
    re_pattern = r'^(페\s?이\s?지\s?:?|시\s?행\s?일\s?자\s?:?|절차서 번호|개\s?정\s?번\s?호\s?:?).*'
    re_pattern_2 = r'.*(페이지(\s?:\s?|\s)\d{1,3}\s?/\s?\d{1,3})$'
    delete_lines = []
    page_delete_lines = {}
    first_header_detect = {}

    def update_target_page(item):
        page_no = item.prov[0].page_no
        target_page.add(page_no)
        if page_no not in page_delete_lines:
            page_delete_lines[page_no] = int(item.prov[0].bbox.b)
        else:
            page_delete_lines[page_no] = min(page_delete_lines[page_no], int(item.prov[0].bbox.b))

    for item, level in result.iterate_items():
        if isinstance(item, TableItem):
            page_no = item.prov[0].page_no
            if page_no in first_header_detect and first_header_detect[page_no]:
                continue
            cnt = 0
            if item.prov[0].bbox.t > page_heights[page_no] * 0.80 and item.prov[0].bbox.b > page_heights[page_no] * 0.5:
                for cell in item.data.table_cells:
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
    delete_line_counter = Counter(delete_lines)
    if len(delete_line_counter.most_common()) > 0:
        delete_line = delete_line_counter.most_common(1)[0][0]
    else:
        delete_line = max_below_line
    return delete_line, target_page, page_delete_lines

def update_target_page_from_text(header_table_delete_line, result, target_page, page_delete_lines, max_height):
    re_pattern_texts = r'^(절\s?차\s?서|품\s?질\s?매\s?뉴\s?얼)$'
    re_pattern = r'^(페\s?이\s?지\s?:?|시\s?행\s?일\s?자\s?:?|절차서 번호).*'
    re_pattern_2 = r'.*(페이지(\s?:\s?|\s)\d{1,3}\s?/\s?\d{1,3})$'
    re_pattern_page_no = r'^\d{1,3}\s?의\s?\d{1,3}$'
    re_pattern_3 = r'.*(페\s?이\s?지\s?:\s?(\d{1,3})?)$'
    re_pattern_3_page_no = r'^\d{1,3}\s?/\s?\d{1,3}$'
    updated_delete_line = header_table_delete_line
    page_num_trigger = {}
    page_delete_line_update = {}
    initial_page_delete_lines = page_delete_lines.copy()

    def update_target_page(item, item_above_flag):
        page_no = item.prov[0].page_no
        if item_above_flag:
            if page_no in page_delete_line_update and page_delete_line_update[page_no] == True:
                pass
            else:
                if page_no in page_delete_lines:
                    del page_delete_lines[page_no]
                    page_delete_line_update[page_no] = True
        target_page.add(item.prov[0].page_no)
        if page_no not in page_delete_lines:
            page_delete_lines[page_no] = int(item.prov[0].bbox.b)
        else:
            page_delete_lines[page_no] = min(page_delete_lines[page_no], int(item.prov[0].bbox.b))

    for item, level in result.iterate_items():
        item_above_flag = False
        if isinstance(item, TextItem):
            page_no = item.prov[0].page_no
            if page_no in initial_page_delete_lines:
                if item.prov[0].bbox.t < initial_page_delete_lines[page_no]:
                    continue
                else:
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
            elif item.prov[0].bbox.t > header_table_delete_line * 0.6:
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
            if re.match(re_pattern, item.text) or re.match(re_pattern_2, item.text) or re.match(re_pattern_3,
                                                                                                item.text):
                if page_no in page_num_trigger and page_num_trigger[page_no]:
                    if re.match(re_pattern_page_no, item.text) or re.match(re_pattern_3_page_no, item.text):
                        update_target_page(item, item_above_flag)
                if item.prov[0].bbox.t < max_height * 0.5:
                    continue
                elif re.match(r'^(절차서 제목)', item.text):
                    continue
                update_target_page(item, item_above_flag)
                page_num_trigger[page_no] = True
    return target_page, page_delete_lines

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
    # print("max_height:", max_height)
    # print("page_header_text:", page_header_text)

    # print("page_heights", page_heights)
    header_table_delete_line, target_page, page_delete_lines = get_delete_line_from_table(max_height, org_document,
                                                                                          page_heights)
    # print("header_table_delete_line:", header_table_delete_line)
    # print("page_delete_lines:", page_delete_lines)
    updated_target_page, updated_page_delete_lines = update_target_page_from_text(header_table_delete_line, org_document,
                                                                                  target_page, page_delete_lines,
                                                                                  max_height)
    # print("updated_target_page:", updated_target_page)
    # print("updated_page_delete_lines:", updated_page_delete_lines)

    # updated_delete_line -= 5

    group_cache = []
    # last_text_item = None
    last_level = 0

    def add_headers(group, new_doc, last_level):
        if group.label == 'page_header':
            new_doc.add_heading(
                text=group.text,
                orig=group.text,
                level=1,
                prov=group.prov[0],
                parent=new_doc.groups[-1]
            )
        elif group.label == 'section_header':
            last_level += 1
            new_doc.add_heading(
                text=group.text,
                orig=group.text,
                level=last_level,
                prov=group.prov[0],
                parent=new_doc.groups[-1]
            )
        elif group.label == 'caption':
            last_level += 1
            new_doc.add_heading(
                text=group.text,
                orig=group.text,
                level=last_level,
                prov=group.prov[0],
                parent=new_doc.groups[-1]
            )

    for key, item in org_document.pages.items():
        if isinstance(item, PageItem):
            # print("item.page_no:", item.page_no)
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
                    middle_line = item.prov[0].bbox.b + (item.prov[0].bbox.t - item.prov[0].bbox.b) / 2
                    if middle_line > updated_page_delete_lines[page_no]:
                        continue
                    else:
                        pass
                    # pass
                else:
                    continue
        if isinstance(item, TableItem):
            if len(group_cache) > 0:
                new_doc.add_group(
                    label=GroupLabel.LIST,
                    name="list",
                    parent=new_doc.body
                )
                for group in group_cache:
                    add_headers(group, new_doc, last_level)
                group_cache = []
                last_level = 0
            if len(new_doc.groups) > 0:
                new_doc.add_table(
                    data=item.data,
                    caption=[],
                    prov=item.prov[0],
                    parent=new_doc.groups[-1]
                )
            else:
                new_doc.add_table(
                    data=item.data,
                    caption=[],
                    prov=item.prov[0],
                    parent=new_doc.body
                )
        if isinstance(item, PictureItem):
            if len(group_cache) > 0:
                new_doc.add_group(
                    label=GroupLabel.LIST,
                    name="list",
                    parent=new_doc.body
                )
                for group in group_cache:
                    last_level = add_headers(group, new_doc, last_level)
                group_cache = []
                last_level = 0
            if len(new_doc.groups) > 0:
                new_doc.add_picture(
                    annotations=[],
                    image=None,
                    caption=[],
                    prov=item.prov[0],
                    parent=new_doc.groups[-1]
                )
            else:
                new_doc.add_picture(
                    annotations=[],
                    image=None,
                    caption=[],
                    prov=item.prov[0],
                    parent=new_doc.body
                )
        if isinstance(item, TextItem):
            if item.label == 'page_header':
                group_cache.append(item)
            elif item.label == 'section_header':
                group_cache.append(item)
            elif item.label == 'caption':
                group_cache.append(item)
            elif item.label == 'list_item':
                if len(group_cache) > 0:
                    new_doc.add_group(
                        label=GroupLabel.LIST,
                        name="list",
                        parent=new_doc.body
                    )
                    for group in group_cache:
                        last_level = add_headers(group, new_doc, last_level)
                    group_cache = []
                    last_level = 0
                if len(new_doc.groups) > 0:
                    new_doc.add_list_item(
                        text=item.text,
                        enumerated=item.enumerated,
                        marker=item.marker,
                        orig=item.text,
                        prov=item.prov[0],
                        parent=new_doc.groups[-1]
                    )
            elif item.label == 'text' or item.label == 'checkbox_unselected' or item.label == 'code' or item.label == 'paragraph':
                if len(group_cache) > 0:
                    new_doc.add_group(
                        label=GroupLabel.LIST,
                        name="list",
                        parent=new_doc.body
                    )
                    for group in group_cache:
                        last_level = add_headers(group, new_doc, last_level)
                    group_cache = []
                    last_level = 0
                if len(new_doc.groups) > 0:
                    new_doc.add_text(
                        label=item.label,
                        text=item.text,
                        orig=item.text,
                        prov=item.prov[0],
                        parent=new_doc.groups[-1]
                    )
                else:
                    new_doc.add_text(
                        label=item.label,
                        text=item.text,
                        orig=item.text,
                        prov=item.prov[0],
                        parent=new_doc.body
                    )

    return new_doc

####################################################
#################### 전처리 코드 #####################
####################################################

#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Chunker implementation leveraging the document structure."""

import logging
import re
from typing import Any, ClassVar, Final, Iterator, Literal, Optional

from pandas import DataFrame
from pydantic import Field, StringConstraints, field_validator
from typing_extensions import Annotated

from docling_core.search.package import VERSION_PATTERN
from docling_core.transforms.chunker import BaseChunk, BaseChunker, BaseMeta
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.document import (
    DocItem,
    DocumentOrigin,
    LevelNumber,
    ListItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel

_VERSION: Final = "1.0.0"

_KEY_SCHEMA_NAME = "schema_name"
_KEY_VERSION = "version"
_KEY_DOC_ITEMS = "doc_items"
_KEY_HEADINGS = "headings"
_KEY_CAPTIONS = "captions"
_KEY_ORIGIN = "origin"

_logger = logging.getLogger(__name__)


class DocMeta(BaseMeta):
    """Data model for Hierarchical Chunker chunk metadata."""

    schema_name: Literal["docling_core.transforms.chunker.DocMeta"] = Field(
        default="docling_core.transforms.chunker.DocMeta",
        alias=_KEY_SCHEMA_NAME,
    )
    version: Annotated[str, StringConstraints(pattern=VERSION_PATTERN, strict=True)] = (
        Field(
            default=_VERSION,
            alias=_KEY_VERSION,
        )
    )
    doc_items: list[DocItem] = Field(
        alias=_KEY_DOC_ITEMS,
        min_length=1,
    )
    headings: Optional[list[str]] = Field(
        default=None,
        alias=_KEY_HEADINGS,
        min_length=1,
    )
    captions: Optional[list[str]] = Field(
        default=None,
        alias=_KEY_CAPTIONS,
        min_length=1,
    )
    origin: Optional[DocumentOrigin] = Field(
        default=None,
        alias=_KEY_ORIGIN,
    )

    excluded_embed: ClassVar[list[str]] = [
        _KEY_SCHEMA_NAME,
        _KEY_VERSION,
        _KEY_DOC_ITEMS,
        _KEY_ORIGIN,
    ]
    excluded_llm: ClassVar[list[str]] = [
        _KEY_SCHEMA_NAME,
        _KEY_VERSION,
        _KEY_DOC_ITEMS,
        _KEY_ORIGIN,
    ]

    @field_validator(_KEY_VERSION)
    @classmethod
    def check_version_is_compatible(cls, v: str) -> str:
        """Check if this meta item version is compatible with current version."""
        current_match = re.match(VERSION_PATTERN, _VERSION)
        doc_match = re.match(VERSION_PATTERN, v)
        if (
            doc_match is None
            or current_match is None
            or doc_match["major"] != current_match["major"]
            or doc_match["minor"] > current_match["minor"]
        ):
            raise ValueError(f"incompatible version {v} with schema version {_VERSION}")
        else:
            return _VERSION


class DocChunk(BaseChunk):
    """Data model for document chunks."""

    meta: DocMeta


class HierarchicalChunker(BaseChunker):
    r"""Chunker implementation leveraging the document layout.

    Args:
        merge_list_items (bool): Whether to merge successive list items.
            Defaults to True.
        delim (str): Delimiter to use for merging text. Defaults to "\n".
    """

    merge_list_items: bool = True

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
                                    for k in sorted(heading_by_level)
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

class GenOSVectorMetaBuilder:
    def __init__(self):
        """빌더 초기화"""
        self.text: Optional[str] = None
        self.n_char: Optional[int] = None
        self.n_word: Optional[int] = None
        self.n_line: Optional[int] = None
        self.i_page: Optional[int] = None
        self.i_chunk_on_page: Optional[int] = None
        self.n_chunk_of_page: Optional[int] = None
        self.i_chunk_on_doc: Optional[int] = None
        self.n_chunk_of_doc: Optional[int] = None
        self.n_page: Optional[int] = None
        self.reg_date: Optional[str] = None
        self.bboxes: str = None

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
        conv_result: ConversionResult = self.converter.convert(file_path, raises_on_error=True)
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
        # chunker: HybridChunker = HybridChunker()
        # TODO: 전처리 코드 넣기
        result = documents

        origin = DocumentOrigin(
            filename=result.origin.filename,
            mimetype="application/pdf",
            binary_hash=result.origin.binary_hash,
        )
        new_doc = DoclingDocument(
            name="file", origin=origin
        )

        max_height, page_header_text, page_heights = get_max_height(result)
        header_table_delete_line, target_page, page_delete_lines = get_delete_line_from_table(max_height, result,
                                                                                              page_heights)
        updated_target_page, updated_page_delete_lines = update_target_page_from_text(header_table_delete_line, result,
                                                                                      target_page, page_delete_lines,
                                                                                      max_height)
        group_cache = None
        last_text_item = None
        for key, item in result.pages.items():
            if isinstance(item, PageItem):
                # print("item.page_no:", item.page_no)
                new_doc.add_page(
                    page_no=item.page_no,
                    size=item.size,
                    image=item.image,
                )
        label_list = ['page_header', 'section_header', 'list_item']
        for item, level in result.iterate_items():
            page_no = item.prov[0].page_no
            if item.prov[0].page_no != 1:
                if isinstance(item, TextItem) and ''.join(item.text.split()) == ''.join(page_header_text.split()):
                    if item.label in label_list:
                        continue
                    elif page_no in updated_page_delete_lines and item.prov[0].bbox.t > updated_page_delete_lines[
                        page_no]:
                        continue
                if page_no in updated_page_delete_lines and item.prov[0].bbox.t > updated_page_delete_lines[page_no]:
                    if item.prov[0].bbox.b < updated_page_delete_lines[page_no]:
                        middle_line = item.prov[0].bbox.b + (item.prov[0].bbox.t - item.prov[0].bbox.b) / 2
                        if middle_line > updated_page_delete_lines[page_no]:
                            continue
                        else:
                            pass
                        # pass
                    else:
                        continue
            if isinstance(item, TableItem):
                #    new_doc.add_table(
                #         data=item.data,
                #         caption=[],
                #         prov=item.prov[0],
                #         parent=new_doc.body
                #     )
                if len(new_doc.groups) > 0:
                    new_doc.add_table(
                        data=item.data,
                        caption=[],
                        prov=item.prov[0],
                        parent=new_doc.groups[-1]
                    )
                else:
                    new_doc.add_table(
                        data=item.data,
                        caption=[],
                        prov=item.prov[0],
                        parent=new_doc.body
                    )
                # if last_text_item:
                #     new_doc.add_group(
                #         label=GroupLabel.LIST,
                #         name="list",
                #         parent=new_doc.body
                #     )
                #     new_doc.add_table(
                #         data=item.data,
                #         caption=[],
                #         prov=item.prov[0],
                #         parent=new_doc.groups[-1]
                #     )
                # else:
                #     new_doc.add_table(
                #         data=item.data,
                #         caption=[],
                #         prov=item.prov[0],
                #         parent=new_doc.body
                #     )
            if isinstance(item, PictureItem):
                new_doc.add_picture(
                    annotations=[],
                    image=None,
                    caption=[],
                    prov=item.prov[0],
                    parent=new_doc.body
                )
            if isinstance(item, TextItem):
                if item.label == 'page_header':
                    new_doc.add_heading(
                        text=item.text,
                        orig=item.text,
                        level=1,
                        prov=item.prov[0],
                        parent=new_doc.body
                    )
                    group_cache = item
                    last_text_item = item
                elif item.label == 'section_header':
                    new_doc.add_heading(
                        text=item.text,
                        orig=item.text,
                        level=item.level,
                        prov=item.prov[0],
                        parent=new_doc.body
                    )
                    group_cache = item
                    last_text_item = item
                elif item.label == 'caption':
                    new_doc.add_heading(
                        text=item.text,
                        orig=item.text,
                        level=1,
                        prov=item.prov[0],
                        parent=new_doc.body
                    )
                    group_cache = item
                    last_text_item = item
                elif item.label == 'list_item':
                    if group_cache != None:
                        new_doc.add_group(
                            label=GroupLabel.LIST,
                            name="list",
                            parent=new_doc.body
                        )
                        group_cache = None
                    if len(new_doc.groups) > 0:
                        new_doc.add_list_item(
                            text=item.text,
                            enumerated=item.enumerated,
                            marker=item.marker,
                            orig=item.text,
                            prov=item.prov[0],
                            parent=new_doc.groups[-1]
                        )
                elif item.label == 'text' or item.label == 'checkbox_unselected' or item.label == 'code' or item.label == 'paragraph':
                    if group_cache != None:
                        new_doc.add_group(
                            label=GroupLabel.LIST,
                            name="list",
                            parent=new_doc.body
                        )
                        group_cache = None
                    if len(new_doc.groups) > 0:
                        new_doc.add_text(
                            label=item.label,
                            text=item.text,
                            orig=item.text,
                            prov=item.prov[0],
                            parent=new_doc.groups[-1]
                        )
                    else:
                        new_doc.add_text(
                            label=item.label,
                            text=item.text,
                            orig=item.text,
                            prov=item.prov[0],
                            parent=new_doc.body
                        )
        # TODO: 최종 전처리 document로 chunk 추출
        chunker: HierarchicalChunker = HierarchicalChunker()
        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=new_doc, **kwargs))
        # TODO: 페이지 관련 이슈 처리 시 해당 부분도 수정 필요
        for chunk in chunks:
            self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks

    def compose_vectors(self, document: DoclingDocument, chunks: List[DocChunk], file_path: str, **kwargs: dict) -> \
            list[dict]:
        pdf_path = file_path.replace('.hwp', '.pdf').replace('.txt', '.pdf').replace('.json', '.pdf')

        if os.path.exists(pdf_path):
            doc = fitz.open(pdf_path)

        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=document.num_pages(),
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z'
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

            vector = (GenOSVectorMetaBuilder()
                      .set_text(chunk.text)
                      .set_page_info(chunk_page, chunk_index_on_page, self.page_chunk_counts[chunk_page])
                      .set_chunk_index(chunk_idx)
                      .set_bboxes(chunk.meta.doc_items[0].prov[0].bbox)
                      .set_global_metadata(**global_metadata)
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

        return vectors

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        document: DoclingDocument = self.load_documents(file_path, **kwargs)
        await assert_cancelled(request)

        # Extract Chunk from DoclingDocument
        chunks: List[DocChunk] = self.split_documents(document, **kwargs)
        await assert_cancelled(request)

        vectors: list[dict] = self.compose_vectors(document, chunks, file_path, **kwargs)
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
