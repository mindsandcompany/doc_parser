from fastapi import Request

from genos_docling_manuals_lms_2_law_html import DocumentProcessor

#file_path = "./input_html_to_pdf/49210_19990828.pdf"
#file_path = "./input/test.html"
file_path = "./input/regulation_html_url/1029.html"

# DocumentProcessor 인스턴스 생성
doc_processor = DocumentProcessor()

# FastAPI 요청 예제
mock_request = Request(scope={"type": "http"})

# 비동기 메서드 실행
import asyncio


async def process_document():
    print(file_path)
    vectors = await doc_processor(mock_request, file_path)
    return vectors


# 메인 루프 실행
result = asyncio.run(process_document())

print(result)