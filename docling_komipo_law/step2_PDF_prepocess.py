import os
import time
from datetime import timedelta
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.base import ImageRefMode
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

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

import json
from pathlib import Path
from collections import Counter
import re

def extract_docling_info(input_dir):
    json_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file == "result.json":
                json_files.append(os.path.join(root, file))

    for idx, file_path in enumerate(json_files, start=1):
        process_pdf(file_path, input_dir)

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

def process_pdf(file_path, input_dir):
    with open(file_path, 'r') as file:
        data = json.load(file)
        org_document = DoclingDocument.model_validate(data)

    output_dir = os.path.dirname(file_path)

    origin = DocumentOrigin(
        filename = org_document.origin.filename,
        mimetype = "application/pdf",
        binary_hash = org_document.origin.binary_hash,
    )

    new_doc = DoclingDocument(
        name = "file", origin=origin
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
            
    

    #new_doc.save_as_json(Path(os.path.join(output_dir, "docling_info_edit.json")))
    with open(os.path.join(output_dir, "result_edit.json"), "w", encoding="utf-8") as fw:
        json.dump(new_doc.export_to_dict(), fw, indent=2, ensure_ascii=False)
    print(f"Processed {file_path}")

input_dir = "./output2/"
extract_docling_info(input_dir)
# input_dir = "./output2/규정_drm해제/절차서"
# extract_docling_info(input_dir)

#input_dir = "./output2/규정_drm해제/매뉴얼/전사매뉴얼-정보-001_사이버위기대응실무매뉴얼_2023-12-19_붙임2_사이버 위기대응 실무매뉴얼_전문.hwp"
#extract_docling_info(input_dir)