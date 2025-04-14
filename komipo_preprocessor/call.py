import os
import json
import asyncio
from rms import DocumentProcessor as RmsProcessor
from standard import DocumentProcessor as StandardProcessor
from fastapi import Request

# DocumentProcessor 인스턴스 생성
rms_processor = RmsProcessor()
standard_processor = StandardProcessor()
mock_request = Request(scope={"type": "http"})  # Mock Request 생성


async def process_document(file_path, type):
    """
    비동기적으로 PDF 파일을 처리하여 결과 반환
    :param file_path: 처리할 PDF 파일 경로
    :return: PDF 처리 결과
    """
    print(f"Processing file: {file_path}")
    vectors = []
    if type == 'rms':
        vectors = await rms_processor(mock_request, file_path)
    elif type == 'standard':
        vectors = await standard_processor(mock_request, file_path)

    return vectors


def save_to_json(data, output_path):
    """
    데이터를 JSON 파일로 저장
    :param data: 저장할 데이터 (list 형태)
    :param output_path: 저장할 JSON 파일 경로
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved JSON: {output_path}")


def get_pdf_files_from_directory(directory_path):
    """
    디렉토리에서 PDF 파일 리스트를 반환
    :param directory_path: 탐색할 디렉토리 경로
    :return: PDF 파일들의 전체 경로 리스트
    """
    pdf_files = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.lower().endswith(".pdf")
    ]
    return pdf_files


async def process_all_pdfs_in_directory(directory_path, output_directory, type):
    """
    디렉토리 내 모든 PDF 파일을 처리하여 각 파일명으로 JSON 저장
    :param directory_path: PDF 파일들이 있는 디렉터리 경로
    :param output_directory: JSON 파일을 저장할 디렉터리 경로
    """
    try:
        # PDF 파일 경로 리스트 가져오기
        pdf_files = get_pdf_files_from_directory(directory_path)

        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # 각 PDF 파일 처리
        for pdf_file in pdf_files:
            try:
                # PDF 파일 처리
                result = await process_document(pdf_file, type)

                # 파일명에서 확장자를 제거한 뒤 JSON 파일 경로 설정
                base_name = os.path.splitext(os.path.basename(pdf_file))[0]
                json_file_path = os.path.join(output_directory, f"{base_name}.json")

                # 결과를 JSON 파일로 저장
                dict_result = [vector.dict() for vector in result]  # 처리 결과 변환
                save_to_json(dict_result, json_file_path)

            except Exception as e:
                print(f"오류 발생! 파일: {pdf_file}, 오류: {e}")
    except Exception as e:
        print(f"전체 프로세스 중 오류 발생: {e}")


# 메인 실행부
if __name__ == "__main__":
    # PDF 파일이 위치한 디렉터리
    input_directory = "D:\workspace\mnc\doc_parser\komipo_preprocessor\doc_rms"

    # JSON 결과를 저장할 디렉터리
    output_directory = "D:\workspace\mnc\doc_parser\komipo_preprocessor\doc_rms"

    # 비동기 실행
    asyncio.run(process_all_pdfs_in_directory(input_directory, output_directory, 'rms'))

    input_directory = "D:\workspace\mnc\doc_parser\komipo_preprocessor\standard"

    # JSON 결과를 저장할 디렉터리
    output_directory = "D:\workspace\mnc\doc_parser\komipo_preprocessor\standard"

    # 비동기 실행
    asyncio.run(process_all_pdfs_in_directory(input_directory, output_directory, 'standard'))
