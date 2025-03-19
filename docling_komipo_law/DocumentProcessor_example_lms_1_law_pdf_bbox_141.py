from fastapi import Request

from genos_docling_manuals_lms_1_law_pdf_bbox_141 import DocumentProcessor

# 파일 경로 및 요청 설정
# file_path = "input/direction01_mis_20140911.pdf"
# file_path = "./01.사규규정지침_drm해제/규정_drm해제/test1/인천복지침-운전-002_2발가스터빈WaterWashing지침_2017-03-15_인천복지침-운전-002(가스터빈 WATER WASHING 지침서(UNIT #2).hwp.pdf"

# ============ 

#file_path = "./01.사규규정지침_drm해제/규정_drm해제/test2/보령이지침-정비-090_OvationSystemSoftware작업절차_2022-08-08_보령2발지침-정비-90 [OVATION SYSTEM SOFTWARE 작업절차].hwp.pdf" # 16739, 로컬에서 정상, 수동적재 정상
# Error on calling Preprocess-api in Autoingestion="지침서테스트-2", run=362 Error: MLOpsServiceException(code=1, errMsg='[preprocess-api] API 실행 중 오류가 발생했습니다. \n url: http://llmops-preprocess-api-service:8080/run/? \n data: {\'document_upsert_id\': 16739} \n e: Caught RuntimeError in replica 0 on device 0.\nOriginal Traceback (most recent call last):\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/parallel/parallel_apply.py", line 96, in _worker\n output = module(*input, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl\n return self._call_impl(*args, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl\n return forward_call(*args, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/easyocr/model/vgg_model.py", line 30, in forward\n contextual_feature = self.SequenceModeling(visual_feature)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl\n return self._call_impl(*args, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl\n return forward_call(*args, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/container.py", line 250, in forward\n input = module(input)\n File

# file_path = "./01.사규규정지침_drm해제/규정_drm해제/test2/신보령지침-운전-011_보일러재순환펌프운전지침서_2018-06-22_신보령지침-운전-011(보일러 재순환펌프 운전지침서).hwp.pdf" # 16895, 33p, 4M, 로컬에서 정상, 수동적재 정상
# Error on calling Preprocess-api in Autoingestion="지침서테스트-2", run=362 Error: MLOpsServiceException(code=1, errMsg='[preprocess-api] API 실행 중 오류가 발생했습니다. \n url: http://llmops-preprocess-api-service:8080/run/? \n data: {\'document_upsert_id\': 16895} \n e: Caught RuntimeError in replica 0 on device 0.\nOriginal Traceback (most recent call last):\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/parallel/parallel_apply.py", line 96, in _worker\n output = module(*input, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl\n return self._call_impl(*args, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl\n return forward_call(*args, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/easyocr/model/vgg_model.py", line 30, in forward\n contextual_feature = self.SequenceModeling(visual_feature)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl\n return self._call_impl(*args, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl\n return forward_call(*args, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/container.py", line 250, in forward\n input = module(input)\n File

# file_path = "./01.사규규정지침_drm해제/규정_drm해제/test2/신서천지침-운전-016_신서천화력보일러재순환펌프(BRP)운전지침서_2021-12-28_신서천지침-운전-016_신서천화력 보일러 재순환펌프(BRP) 운전지침서.hwp.pdf" # 16934 , 로컬에서 정상, 시간이 많이 걸림. 41p, 12M, 수동적재 정상
# Error on calling Preprocess-api in Autoingestion="지침서테스트-2", run=362 Error: MLOpsServiceException(code=1, errMsg='[preprocess-api] API 실행 중 오류가 발생했습니다. \n url: http://llmops-preprocess-api-service:8080/run/? \n data: {\'document_upsert_id\': 16934} \n e: Caught RuntimeError in replica 0 on device 0.\nOriginal Traceback (most recent call last):\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/parallel/parallel_apply.py", line 96, in _worker\n output = module(*input, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl\n return self._call_impl(*args, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl\n return forward_call(*args, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/easyocr/model/vgg_model.py", line 30, in forward\n contextual_feature = self.SequenceModeling(visual_feature)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl\n return self._call_impl(*args, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl\n return forward_call(*args, **kwargs)\n File "/app/.venv/lib/python3.10/site-packages/torch/nn/modules/container.py", line 250, in forward\n input = module(input)\n File

##file_path = "./01.사규규정지침_drm해제/규정_drm해제/test2/전사절차-정보-001_정보시스템업무절차서_2024-10-14_전사절차-정보-001_정보시스템업무절차서_[서식7-8] EUC 파일 백업대장.hwp.pdf" # 16451, 로컬에서 오류는 없으나 빈값 출력, 수동적재 정상
# Error on calling Preprocess-api in Autoingestion="지침서테스트-2", run=362 Error: MLOpsServiceException(code=1, errMsg='[preprocess-api] API 실행 중 오류가 발생했습니다. \n url: http://llmops-preprocess-api-service:8080/run/? \n data: {\'document_upsert_id\': 16451} \n e: (asyncmy.errors.OperationalError) (1364, "Field \'data_type\' doesn\'t have a default value")\n[SQL: INSERT INTO vector_ids (is_active, reg_date, mod_date) VALUES (%s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) RETURNING vector_ids.id]\n[parameters: (True,)]\n(Background on this error at: https://sqlalche.me/e/20/e3q8)

# file_path = "./01.사규규정지침_drm해제/규정_drm해제/test2/보령삼지침-운전-034_7,8호기냉간기동지침서_2014-12-05_보령삼지침-운전-034 냉간기동 지침서_REV01.hwp.pdf" # 17162, (1, cancelled), 로컬에서 정상. 수동적재 오류

# =====================
# 22091, 보령이지침-운전-001_Unit비정상상황시조치_2023-11-13_3. Unit 비정상 상황 시 조치(107항목).hwp.pdf  --> 수동적재 정상, 벡터 1000 개 이상
# 23526, 전사절차-정보-006_설비마스터표준화작업절차_2024-07-09_설비마스터 표준화 작업 절차.hwp.pdf  --> 수동적재 정상, 벡터 1000 개 이상
# 24311, 보령3발지침-운전-032_7,8호기소방설비운전지침서_2020-03-30_부록 7-10 J-71560-ZM-105-001.pdf.pdf
# 24332, 보령3발지침-운전-032_7,8호기소방설비운전지침서_2020-03-30_부록 7-10 J-71560-ZM-105-001.pdf.pdf -- 수동적재 오류, 벡터 0개.
# 24542, 보령3발지침-운전-032_7,8호기소방설비운전지침서_2020-03-30_부록 7-11 J-71560-ZM-105-002.pdf.pdf
# file_path = "./01.사규규정지침_drm해제/규정_drm해제/test3/보령3발지침-운전-032_7,8호기소방설비운전지침서_2020-03-30_부록 7-11 J-71560-ZM-105-002.pdf.pdf"
# 24554, 보령3발지침-운전-032_7,8호기소방설비운전지침서_2020-03-30_부록 7-11 J-71560-ZM-105-002.pdf.pdf -- 수동적재 오류, 벡터 0개.
# 24941이후 적재 멈춤. 

## ===========================
#file_path = "./01.사규규정지침_drm해제/규정_drm해제/test4/신보령지침-운영-001_기동용연료유하역작업_2023-11-15_기동용 연료유 하역작업 지침서 전문.hwp.pdf"

#file_path = "./input/Invalid_Code_Point/신보령지침-정비-036_보조터빈DXDGapSetting및교_2019-04-18_두산중공업 BFPT DXD SETTING 지침서.pdf.pdf"
#file_path = "./input/mimetype_test_plain/전사절차-안전보건-035_AEO수출입안전관리절차서_2020-05-18_AEO 수출입안전관리 절차서 NO.3 운영.pdf.pdf"
#file_path = "./input/cancelled/발전소공용지침-운영-001_기력및복합발전소성능시험지침서_2021-05-11_[발전소공용지침-운영-001]기력 및 복합 발전소 성능시험 지침서_REV1.pdf.pdf"
#file_path = "./input/manual/서울매뉴얼-안전-001_서울안전보건경영매뉴얼_2023-04-03_0. [서울매뉴얼-안전-001] 안전보건 경영매뉴얼.hwp.pdf"
#file_path = "./input/Invalid_Code_Point/신보령지침-정비-036_보조터빈DXDGapSetting및교_2019-04-18_두산중공업 BFPT DXD SETTING 지침서.pdf.pdf"
file_path = "./input/mimetype_test_plain/건공절차-건설-001_건설기자재공장검사업무절차_2020-10-21_서식7. 신재생 발전설비 표준 QIP&ITP.pdf.pdf"

# DocumentProcessor 인스턴스 생성
doc_processor = DocumentProcessor()

# FastAPI 요청 예제
mock_request = Request(scope={"type": "http"})

# 비동기 메서드 실행
import asyncio
#import json
import pprint

async def process_document():
    print(file_path)
    vectors = await doc_processor(mock_request, file_path)
    return vectors


# 메인 루프 실행
result = asyncio.run(process_document())

#print(result)

with open("output_pdf_bbox.txt", "w", encoding="utf-8") as f:
    #print(result, file=f)
    #f.write(json.dumps(result, indent=4, ensure_ascii=False))
    f.write(pprint.pformat(result, indent=4))
