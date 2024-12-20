from fastapi import Request

from genos_docling_manuals import DocumentProcessor

# 파일 경로 및 요청 설정
file_path = "resources/direction01_mis_20140911.pdf"

# DocumentProcessor 인스턴스 생성
doc_processor = DocumentProcessor()

# FastAPI 요청 예제
mock_request = Request(scope={"type": "http"})

# 비동기 메서드 실행
import asyncio


async def process_document():
    vectors = await doc_processor(mock_request, file_path)
    return vectors


# 메인 루프 실행
result = asyncio.run(process_document())

print(result)