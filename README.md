# Rag Project

### Language
- Python 3.12

### Directory
- /comm : .env 파일등 공통으로 사용할 데이터 
- /milvus : rag service에 사용할 milvus
- /py12 : rag service에 적용되는 파이썬 데이터
- /experiment : rag나 agent 등 성능 비교 실험

### Experiment
#### 1-병렬처리 실험
* FastAPI 환경에서 동기 -> 비동기 -> 멀티 스레드 실험
#### 2-rag최적화 실험
* 동일한 질문에 따른 리트리버 비교 실험
#### 3-agent실험
* LLM모델이 질문에 따른 tool을 선택하는지 실험
#### 4-vlm graph 실험
* 질문시 이미지가 있는경우 OCR+VLM 실행, lang graph 실험
#### 5-mcp 실험
* 질문에 따라 LLM이 여러 도구 사용하거나 단일 도구 사용하여 답변