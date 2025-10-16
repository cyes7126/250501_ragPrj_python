from fastapi import FastAPI, Request
import asyncio
import time
import math
import nest_asyncio
import uvicorn

app = FastAPI()

# 사용할 스레드 수
MAX_CONCURRENT = 50
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# 파일 임베딩 API
@app.post("/embed")
async def run_heavy_task(pages: int = 1000):
    start = time.time()
    print(f"{time.strftime("%H:%M:%S", time.localtime(start))} embed_page-{pages} 요청!")

    # 1. 동기 함수
    #embed_page_all(total_page=pages)
    
    # 2. 비동기 함수(단일 스레드)
    #await embed_page_async(total_page=pages)
    
    
    # 3. 비동기 함수(멀티 스레드) - semaphore로 관리
    range_length = max(10, math.ceil(pages / MAX_CONCURRENT))
    page_ranges = [
        (i, min(i + range_length - 1, pages - 1))
        for i in range(0, pages, range_length)
    ]

    tasks = [
        handle_limited_thread(
            embed_page_async_multi,
            total_page=pages,
            start=start,
            end=end,
        )
        for start, end in page_ranges
    ]
    await asyncio.gather(*tasks)

    end = time.time()
    print(f"{time.strftime("%H:%M:%S", time.localtime(end))} embed_page-{pages} 종료!")
    return {
        "message": f"EMBED-{pages} 작업 완료",
        "duration": round(end - start, 2)
    }


# 임베딩 삭제 API
@app.post("/del")
async def run_light_task(pages: int = 1000):
    start = time.time()
    print(f"{time.strftime("%H:%M:%S", time.localtime(start))} del-{pages} 요청! {start}")

    # 1. 동기 함수
    #del_embed_all(total_page=pages)
    
    # 2. 비동기 함수
    #del_embed_async(total_page=pages)

    # 3. 비동기 함수(스레드) - semaphore로 관리
    await handle_limited_thread(del_embed_async, total_page=pages)

    end = time.time()
    print(f"{time.strftime("%H:%M:%S", time.localtime(end))} del-{pages} 종료!")
    return {
        "message": f"DELETE-{pages} 작업 완료",
        "duration": round(end - start, 2)
    }

async def handle_limited_thread(func, *args, **kwargs):
    async with semaphore:
        result = await func(*args, **kwargs)
        return result


# 동기 임베딩
def embed_page_all(total_page):
    print(f"{time.strftime("%H:%M:%S", time.localtime(time.time()))} --- embed_page-{total_page} 시작")
    # 10페이지당 20초로 처리 - 파일 접근 등 전/후 처리에 소요되는 시간을 감안
    process_time = math.floor((total_page/10)*20)
    print(f"{time.strftime("%H:%M:%S", time.localtime(time.time()))} --- embed_page-{total_page} - 작업시간 {process_time} ")
    time.sleep(process_time)
    print(f"{time.strftime("%H:%M:%S", time.localtime(time.time()))} --- embed_page-{total_page} 종료")

# 비동기 임베딩 - 단일 스레드(총 페이지 한번에 처리)
async def embed_page_async(total_page):
    print(f"{time.strftime("%H:%M:%S", time.localtime(time.time()))} --- embed_page-{total_page} 시작")
    # 10페이지당 20초로 처리 - 파일 접근 등 전/후 처리에 소요되는 시간을 감안
    process_time = math.floor((total_page/10)*20)
    print(f"{time.strftime("%H:%M:%S", time.localtime(time.time()))} --- embed_page-{total_page} - 작업시간 {process_time} ")
    await asyncio.sleep(process_time)
    print(f"{time.strftime("%H:%M:%S", time.localtime(time.time()))} --- embed_page-{total_page} 종료")

# 비동기 임베딩 - 멀티 스레드(range 범위를 잡아 처리)
async def embed_page_async_multi(total_page, start, end):
    print(f"{time.strftime("%H:%M:%S", time.localtime(time.time()))} --- embed_page-{total_page}. {start}~{end} 시작")
    # page_range당 30초로 처리
    await asyncio.sleep(30)
    print(f"{time.strftime("%H:%M:%S", time.localtime(time.time()))} --- embed_page-{total_page}. {start}~{end} 종료")

# 동기 임베딩 삭제
def del_embed_all(total_page):
    print(f"{time.strftime("%H:%M:%S", time.localtime(time.time()))} --- del-{total_page} 시작")
    time.sleep(40)
    print(f"{time.strftime("%H:%M:%S", time.localtime(time.time()))} --- del-{total_page} 종료")

# 비동기 임베딩 삭제
async def del_embed_async(total_page):
    print(f"{time.strftime("%H:%M:%S", time.localtime(time.time()))} --- del-{total_page} 시작")
    await asyncio.sleep(40)
    print(f"{time.strftime("%H:%M:%S", time.localtime(time.time()))} --- del-{total_page} 종료")


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    now = time.time()
    print(f"{time.strftime('%H:%M:%S', time.localtime(now))} ▶ [요청 수신] {request.method} {request.url}")
    response = await call_next(request)
    return response

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=False, access_log=True)
