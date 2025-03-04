import os
from pathlib import Path

from docling_core.types.doc import ImageRefMode

from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend

from docling.datamodel.pipeline_options import PdfPipelineOptions

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_core.types.doc import (
    PictureItem
)

def list_files_recursively(root_dir: Path) -> list[str]:
    file_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    return file_paths

import json
from docling_core.types.doc import (
    PageItem
)

if __name__ == "__main__":
    target_directory = Path("./input")
    # target_directory = Path("./01.사규규정지침_drm해제/규정_drm해제")
    output_directory = Path("./output")
    ## 파일 목록 확인
    files: list[str] = list_files_recursively(target_directory)
    pipe_line_options = PdfPipelineOptions()
    pipe_line_options.generate_page_images = True
    # pipe_line_options.generate_table_images = True
    pipe_line_options.generate_picture_images = True
    pipe_line_options.do_ocr = True
    pipe_line_options.ocr_options.lang = ["ko", 'en']
    pipe_line_options.do_table_structure = True
    pipe_line_options.images_scale = 1
    pipe_line_options.table_structure_options.do_cell_matching = True
    pipe_line_options.ocr_options.model_storage_directory = "/home/mnc/temp/.EasyOCR/model"

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipe_line_options,
                backend=DoclingParseV2DocumentBackend
            )
        }
    )

    for file in files:
        print(file)
        result = converter.convert(file, raises_on_error=True)
        input_doc = result.input
        print(input_doc)
        print(input_doc.file)
        path = input_doc.file
        page_count = input_doc.page_count
        back = DoclingParseV2DocumentBackend(
           input_doc,
           Path(file)
        )
        output_path = output_directory.joinpath(*path.parts[1:-1], path.stem)
        print("output_path:", output_path)
        print("output_path type:", type(output_path))
        output_path.mkdir(parents=True, exist_ok=True)
        for i in range(page_count):
           page = back.load_page(i)
           img = page.get_page_image(2)
           img.save(f"{output_path}/{i}.png")
        # _image_to_hexhash()
        picture_uri_cache = {}
        #for item, level in result.document.iterate_items():
        #    if isinstance(item, PictureItem):
        #        picture_uri_cache[item.self_ref] = item._image_to_hexhash()
        # with open(f"{output_path}/result.json", "w", encoding="utf-8") as fw:
        #     doc_dict = result.document.export_to_dict()
        #     # for key, page in doc_dict["pages"].items():
        #     #     page["image"]["uri"] = ""
        #     # for picture in doc_dict["pictures"]:
        #     #     picture["image"]["uri"] = ""
        #     # for key, picture in doc_dict["pictures"].items():
        #     #     picture["image"]["uri"] = 
        #     json.dump(doc_dict, fw, indent=2, ensure_ascii=False)
        # result.document._make_copy_with_refmode(image_mode=ImageRefMode.PLACEHOLDER, artifacts_dir=Path(f"{output_path}/artifacts"))
        
        artifacts_dir=Path(f"{output_path}/artifacts")
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = artifacts_dir.parent
        result_img = result.document._with_pictures_refs(image_dir=artifacts_dir, reference_path=reference_path)
        for item, level in result_img.iterate_items():
            if isinstance(item, PictureItem):
                #picture_uri_cache[item.self_ref] = item._image_to_hexhash()
                picture_uri_cache[item.self_ref] = item.image.uri
        with open(f"{output_path}/result.json", "w", encoding="utf-8") as fw:
            doc_dict = result_img.export_to_dict()
            for key, page in doc_dict["pages"].items():
                page["image"]["uri"] = ""
            json.dump(doc_dict, fw, indent=2, ensure_ascii=False)
        #result.document.save_as_json(
        #    filename=Path(f"{output_path}/result.json"),
        #    image_mode=ImageRefMode.REFERENCED,
        #    artifacts_dir=Path(f"{output_path}/artifacts"),
        #)
        print('Processed,, ', output_path)
        print(picture_uri_cache)
