from fastapi import Request

from genos_docling_manuals_lms_3_klaw import DocumentProcessor

#file_path = "./input_html_to_pdf/49210_19990828.pdf"
# file_path = "./input/49210_19990828.html"
# file_path = "./input_html/266391_20241112.html"
# file_path = "./input_html/266391_20241112.html"
file_path = "./input/klaw_2501_url/3274_19890329.html"

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