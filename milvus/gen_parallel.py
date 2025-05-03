'''
Threading	I/O에 강함, 구현 쉬움	GIL 때문에 CPU 작업에 부적합	웹 요청, 파일 업로드 등
asyncio	매우 가볍고 빠름	CPU 작업엔 부적합	API 호출, 비동기 응답
Multiprocessing	CPU 병렬처리 가능	메모리 많이 쓰고 느림	PDF 파싱, 이미지 처리 등
Cluster (Celery 등)	대규모 병렬 작업 가능	셋업 복잡	대량 파일 처리, RAG 시스템 등
'''

# pdfplumber, multiprocessing : 비동기 처리안됨. blocking. 시작하면 끝까지 기다려야함.
# Celery - 비동기 구조

from multiprocessing import Pool, cpu_count
import pdfplumber
import pdf_marker  # 예시
from chromadb import Client  # 예시

def extract_text_from_page(page, pdf_path):
    # pdfplumber로 기본 추출
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[page].extract_text()

    if is_weird(text):
        # pdf-marker 사용
        text = pdf_marker.read(pdf_path, page)
        if is_weird(text):
            text = pdf_marker.read(pdf_path, page, use_ocr=True)

    return page, text

def process_page_range(page_range, pdf_path):
    results = []
    for page in page_range:
        page_num, text = extract_text_from_page(page, pdf_path)
        results.append((page_num, text))
    return results

def split_ranges(total_pages, chunk_size=10):
    return [range(i, min(i + chunk_size, total_pages)) for i in range(0, total_pages, chunk_size)]

if __name__ == "__main__":
    import nest_asyncio
    import uvicorn
    nest_asyncio.apply()

    pdf_path = "input.pdf"
    total_pages = 1000  # 또는 코드로 계산
    page_ranges = split_ranges(total_pages, chunk_size=10)

    with Pool(processes=cpu_count()) as pool:
        all_texts_nested = pool.starmap(process_page_range, [(r, pdf_path) for r in page_ranges])

    # 평탄화
    all_texts = [text for sublist in all_texts_nested for (_, text) in sublist]

    # 8. 청크 분할
    chunks = split_into_chunks(all_texts, chunk_size=500)

    # 9. Chroma에 저장
    client = Client()
    collection = client.get_or_create_collection("pdf_chunks")
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[f"chunk_{i}"])
