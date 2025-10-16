from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from mcp.server.fastmcp import FastMCP
from langchain_tavily import TavilySearch
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

base_path = './'
env = f'{base_path}/../../comm/.env'
load_dotenv(env)

# MCP 서버 인스턴스 생성
mcp = FastMCP("mcpServer")
llm_model = "gpt-4o-mini"

def rough_schedule_to_fixed(rough_schedule: str)->str:
    """대략적인 여행시기를 기준으로 명확한 여행년월을 계산하여 반환"""
    now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y년 %m월")
    chat_model = ChatOpenAI(model=llm_model)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """당신은 입력된 정보에 따라 명확한 년월을 반환해주는 도우미입니다. 
            현재 날짜에 계산할 값을 적용하여, 명확한 년월 값을 답변해주세요. 
            답변 예시 : 2000년 1월"""
        ),
        (
            "human",
            f'현재 날짜 : {now}, 계산할 값 : {rough_schedule}'
        )
    ])
    chain = prompt | chat_model
    response = chain.invoke({})
    return response.content

# MCP - 축제, 전시회 검색
@mcp.tool()
def get_events(destination: str, rough_schedule: str)-> list:
    """도착지와 대략적인 여행일절을 받아 도착지 주변 전시회나 축제 등 이벤트 반환.
    rough_schedule 예시: 다음주, 내년"""
    plan_ym = rough_schedule_to_fixed(rough_schedule)
    search_tools = TavilySearch(
        max_results=3,
        topic="general",
    )

    # 전시회 검색
    query = f"{plan_ym} {destination} 전시회"
    res = search_tools.invoke({"query": query})
    result_list = res.get("results", [])

    # 축제 검색
    query = f"{plan_ym} {destination} 축제"
    res = search_tools.invoke({"query": query})
    result_list.extend(res.get("results", []))

    return [{'title':r['title'], 'url':r['url']} for r in result_list]

# MCP 도구 등록 - 종합 브리핑 도구(다른 도구를 순차적으로 호출)
@mcp.tool()
def info_tour() -> str:
    """사용자의 여행 정보 생성"""
    return """
    다음을 순서대로 실행하고, 실행한 결과를 사용자에게 알려주세요.
    1. maps_directions 도구를 활용하여 출발지에서 도착지까지 이동하는 방법과 시간을 알려주세요. 도착지, 출발지를 모른다면 사용자에게 질문하세요.
    2. get_weather 도구를 활용하여 도착지의 날씨를 출력합니다. 
    3. get_events 도구를 사용하여 여행시기를 기준으로 도착지의 전시회, 축제를 출력합니다. 여행시기에 대해 아예 언급이 없을때만 사용자에게 질문하세요. 
    4. map_search_places 도구를 사용하여 도착지 주변의 맛집, 카페 정보를 출력합니다.
    
    출력은 다음과 같이 해주세요.
    ## 사용자님을 위한 여행 정보입니다.
    
    ### 이동 방법
    [maps_directions의 결과]
    
    ### 도착지 날씨
    [get_weather의 결과]
    
    ### 도착지 주변 전시회 및 축제
    [get_events의 결과] (링크를 함께 제공합니다)

    ### 도착지 주변 맛집 및 카페
    [map_search_places의 결과]
    """

if __name__ == "__main__":
    # mcp 서버 실행(http 스트리밍 모드, 포트 8000)
    mcp.run(transport="streamable-http")

