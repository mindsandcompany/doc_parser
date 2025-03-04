# Image Viewer 설치
### 폐쇄망 환경일 경우
- web/ 폴더의 node.tar.gz, yarn.tar.gz 설치 : 설치방법은 web/폴더의 note.txt 참조
### 인터넷 환경일 경우
- node-v22.11.0 설치
- yarn-v1.22.22 설치
- npm install 또는 yarn install

# Image Viewer 실행
- target 폴더(전처리의 output폴더) pdf문서가 전처리가 완료된 상태로 준비되어야 함.
- 필요시 아래 target 폴더 위치 조정(server.js)
    ```
    const FOLDER_TO_SERVE = path.join(__dirname, '../../output');
    const FOLDER_TO_SERVE_JSON = path.join(__dirname, '../../output');
    ```
- server.js 실행
    ```
    web/image-viewer-2-3/node server.js
    ```
- bbox color
  - Hybrid chunk : violet
  - Hierachical Chunk : red
  - items(texts, tables, pictures) : orange

# PDF 전처리
- step1, step2, step3, step4 순서대로 실행
- 필요시 step1의 input 폴더 위치 조정
# 전처리 코드 설명
- step1 : PDF파일을 Docling으로 result.json, 페이지이미지 추출
- step2 : result.json 을 읽어서, 전처리(헤더 날리는 등)후 result_edit.json 생성.
- step3 : result_edit.json 을 읽어서, hierachical Chunk 생성. result_chunks.json생성.
- step4 : result_edit.json 을 읽어서, hybrid Chunk 생성. result_chunks_hybrid.json 생성
  - step3, step4 는 같은 코드이나, 470 라인, hierachical/hybrid 설정만 다름.
