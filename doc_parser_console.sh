#!/bin/bash

# 입력 인자로 파일 경로를 전달받기
FILE_PATH="$1"
FROM=pdf
TO=md
OCR_LANG=ko,en
OUTPUT=./parsed_output

# 파일 경로가 제공되지 않은 경우 에러 메시지 출력
if [ -z "$FILE_PATH" ]; then
  echo "에러: 변환할 PDF 파일 경로를 입력하세요."
  echo "사용법: ./doc_parser_console.sh <파일 경로>"
  exit 1
fi


# docling 명령어 실행
docling \
  --from $FROM \
  --to $TO \
  --ocr \
  --ocr-lang $OCR_LANG \
  --output $OUTPUT \
  "$FILE_PATH"
