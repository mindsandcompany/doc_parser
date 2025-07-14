from test import DocumentProcessor
from fastapi import Request
import asyncio

file_path = '/Users/namseunghyun/workspace/doc_parser/preprocessor_code/기동_Cold_Start_up__절차서_110429260.pdf'

preprocessor = DocumentProcessor()

async def preprocessor_call(request: Request, file_path: str):
    return await preprocessor(None, file_path)

if __name__ == "__main__":
    # asyncio.run으로 async 작업 실행
    asyncio.run(preprocessor_call(None, file_path))