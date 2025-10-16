import uvicorn
import os
import json

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from dotenv import load_dotenv

base_path = './'
env = f'{base_path}/../../comm/.env'
load_dotenv(env)
SHARE_PATH = "../../share/upload"
llm_model = "gpt-4o-mini"

def load_mcp_config():
    try:
        with open("./mcp_config.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"설정 파일을 읽는 중 오류 발생: {str(e)}")
        return None

def create_server_config():
    config = load_mcp_config()
    server_config = {}

    if config and "mcpServers" in config:
        for server_name, server_config_data in config["mcpServers"].items():
            print(f"mcpServer config 조회 : {server_name} {server_config_data}")
            if "command" in server_config_data: # command 있으면 stdio방식
                server_config[server_name] = {
                    "command": server_config_data.get("command"),
                    "args": server_config_data.get("args", []),
                    "transport": "stdio"
                }
            elif "url" in server_config_data: # url이 있으면 sse
                server_config[server_name] = {
                    "url": server_config_data.get("url"),
                    "transport": "sse"
                }
    return server_config

# 에이전트를 위한 프롬프트 템플릿 생성
def create_prompt_template() -> ChatPromptTemplate:
    """에이전트를 위한 프롬프트 템플릿을 생성"""
    system_prompt = """
    당신은 친절하고 도움이 되는 AI 여행 도우미 "떠나봇" 입니다.
    출발지, 도착지, 대략적인 여행시기를 안다면 바로 info_tour() 도구의 지침을 따르면 됩니다. 
    아니라면, 필요한 정보를 물어보고 나서 도구의 지침을 따릅니다. 

    다음과 같은 도구들을 활용하여 사용자를 도와드릴 수 있습니다:
    - 출발지, 도착지, 대략적인 여행시기를 기준으로 여행 정보를 생성합니다.
    - 출발지와 도착지를 기준으로 이동경로를 가져올 수 있습니다.
    - 도착지의 날씨를 가져올 수 있습니다.
    - 도착지와 대략적인 여행시기를 기준으로 전시회나 축제 정보를 가져올 수 있습니다.
    - 도착지 주변의 맛집,카페 정보를 가져올 수 있습니다.
    

    사용자와의 대화에서 다음 원칙을 지켜주세요:
    1. 항상 친절하고 정중한 태도로 응답해주세요.
    2. 사용자의 질문을 정확히 이해하고 관련된 도구를 적절히 활용해주세요.
    3. 응답은 명확하고 이해하기 쉽게 구성해주세요.
    4. 필요 시 추가 정보나 설명을 제공하여 사용자에게 더 나은 도움을 주세요.
    """

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])


# ReAct 에이전트 생성을 위한 함수
def create_agent(tools):
    """주어진 도구를 사용하여 에이전트를 생성"""
    memory = InMemorySaver()  # 대화이력을 메모리에 저장
    prompt = create_prompt_template()
    llm = ChatOpenAI(model=llm_model, streaming=False)
    # ReAct 에이전트 생성
    return create_react_agent(llm, tools, checkpointer=memory, prompt=prompt)


# FastAPI lifespan을 사용한 에이전트 준비
## lifespan : 애플리케이션 시작 전 후 동작을 제어. app시작적에 MCP 서버를 연결하고, 도구를 로드후 에이전트 생성
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 애플리케이션의 생명주기 동안 MCP 연결 및 에이전트 설정을 관리"""
    print("애플리케이션 시작: MCP 서버에 연결하고 에이전트 설정 시작")
    # streamablehttp_client : mcp sdk 제공 함수. streamablehttp 서버의 클라이언트 생성
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            # local tools
            # session으로 도구리스트를 얻음
            await session.initialize()
            tools = await load_mcp_tools(session)

            # smithery tools
            multi_client = MultiServerMCPClient(create_server_config())
            tools.extend(await multi_client.get_tools())
            app.state.agent_executor = create_agent(tools)
            print("에이전트 설정 완료. 애플리케이션이 준비되었습니다.")
            yield
    print("애플리케이션 종료")
    # app 종료시엔 연결을 정리
    app.state.agent_executor = None


# FastAPI 앱 인스턴스 생성 및 정적 파일 설정
# lifespan 관리자 사용
app = FastAPI(lifespan=lifespan)

# 정적 파일 마운트
static_path = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

templates_path = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))


# 메인 페이지 라우트
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """메인 채팅 페이지 랜더링"""
    return templates.TemplateResponse("index.html", {"request": request})


# 에이전트 응답 스트리밍 함수
async def stream_agent_response(agent_executor, message: str, session_id: str):
    """에이전트의 응답을 스트리밍하는 비동기 제너레이터"""
    if agent_executor is None:
        yield "에이전트가 아직 준비중.. 잠시후 다시 시도해주세요."
        return

    try:
        # 히스토리 기록을 위해 세션아이디 설정
        config = {"configurable": {"thread_id": session_id}}
        input_message = HumanMessage(content=message)

        # astream_events 사용하여 응답 스트리밍
        async for event in agent_executor.astream_events(
                {"messages": [input_message]},
                config=config,
                version="v1"
        ):
            kind = event["event"]
            # 텍스트 응답을 추출하여 클라이언트로 전송
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # 스트리밍된 컨텐츠를 클라이언트로 전송
                    yield content
            elif kind == "on_tool_start":
                # 도구 사용 시작을 클라이언트에 알림
                print(f"도구 사용 시작: {event['name']}")
            elif kind == "on_tool_end":
                # 도구 사용 완료를 클라이언트에 알림
                print(f"도구 사용 완료: {event['name']}")

    except Exception as e:
        print(f"응답 중 오류 발생: {str(e)}")
        yield f"오류가 발생했습니다: {str(e)}"


# 채팅 API 엔드포인트
@app.post("/chat")
async def chat(request: Request,
               message: str = Form(...),
               session_id: str = Form(...)):
    """사용자 메시지를 받아 에이전트의 응답을 스트리밍"""
    agent_executor = request.app.state.agent_executor
    return StreamingResponse(
        stream_agent_response(agent_executor, message, session_id),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)