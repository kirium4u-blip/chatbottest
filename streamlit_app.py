import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI as ChatOpenAI_OAI
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
import tempfile
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool

# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ✅ SerpAPI 검색 툴 정의
def search_web():
    search = SerpAPIWrapper()
    def run_with_source(query: str) -> str:
        results = search.results(query)
        organic = results.get("organic_results", [])
        formatted = []
        for r in organic[:5]:
            title = r.get("title")
            link = r.get("link")
            source = r.get("source")
            snippet = r.get("snippet")  # ✅ snippet 추가
            if link:
                formatted.append(f"- [{title}]({link}) ({source})\n  {snippet}")
            else:
                formatted.append(f"- {title} (출처: {source})\n  {snippet}")
        return "\n".join(formatted) if formatted else "검색 결과가 없습니다."
    return Tool(
        name="web_search",
        func=run_with_source,
        description="실시간 뉴스 및 웹 정보를 검색할 때 사용합니다. 결과는 제목+출처+링크+간단요약(snippet) 형태로 반환됩니다."
    )

# ✅ PDF 업로드 → 벡터DB → 검색 툴 생성
def load_pdf_files(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)

    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Use this tool to search information from the pdf document"
    )
    return retriever_tool

# ✅ Agent 대화 실행
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    return result['output']

# ✅ 세션별 히스토리 관리
def get_session_history(session_ids):
    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}
    if session_ids not in st.session_state["session_history"]:
        st.session_state["session_history"][session_ids] = ChatMessageHistory()
    return st.session_state["session_history"][session_ids]

# ✅ 이전 메시지 출력
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])

# ✅ 메인 실행
def main():
    st.set_page_config(page_title="AI 비서", layout="wide", page_icon="🤖")

    with st.container():
        # 필요 시 경로 조정
        # st.image('./chatbot_logo.png', use_container_width=True)
        st.markdown('---')
        st.title("안녕하세요! RAG를 활용한 'AI 비서 톡톡이' 입니다")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}

    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input("OPENAI API 키", placeholder="Enter Your API Key", type="password")
        st.session_state["SERPAPI_API"] = st.text_input("SERPAPI_API 키", placeholder="Enter Your API Key", type="password")
        # ★ 변경: 단일 질문 모드 토글 (기본 ON)
        single_turn = st.checkbox("단일 질문 모드 (매 질문마다 이전 대화 숨기기)", value=True)
        st.markdown('---')
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, key="pdf_uploader")

    # ✅ 키 입력 확인
    if st.session_state["OPENAI_API"] and st.session_state["SERPAPI_API"]:
        os.environ['OPENAI_API_KEY'] = st.session_state["OPENAI_API"]
        os.environ['SERPAPI_API_KEY'] = st.session_state["SERPAPI_API"]

        # 도구 정의
        tools = []
        if pdf_docs:
            pdf_search = load_pdf_files(pdf_docs)
            tools.append(pdf_search)
        tools.append(search_web())

        # LLM 설정
        # 주의: 위에서 chat_models의 ChatOpenAI와 langchain_openai의 ChatOpenAI 이름이 같아 충돌 가능
        # 여기서는 langchain_openai 쪽 별칭을 명시적으로 사용
        llm = ChatOpenAI_OAI(model_name="gpt-4o-mini", temperature=0)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                "Be sure to answer in Korean. You are a helpful assistant. "
                "Make sure to use the `pdf_search` tool for searching information from the pdf document. "
                "If you can't find the information from the PDF document, use the `web_search` tool for searching information from the web. "
                "If the user’s question contains words like '최신', '현재', or '오늘', you must ALWAYS use the `web_search` tool to ensure real-time information is retrieved. "
                "Please always include emojis in your responses with a friendly tone. "
                "Your name is `AI 비서 톡톡이`. Please introduce yourself at the beginning of the conversation."),
                ("placeholder", "{chat_history}"),
                ("human", "{input} \n\n Be sure to include emoji in your responses."),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # 입력창
        user_input = st.chat_input('질문이 무엇인가요?')

        if user_input:
            # ★ 변경: 단일 질문 모드일 경우 이전 메시지/히스토리 초기화
            if single_turn:
                st.session_state["messages"] = []              # 화면 채팅 초기화
                st.session_state["session_history"] = {}        # 히스토리 초기화

            session_id = "default_session"
            session_history = get_session_history(session_id)

            # 누적 모드일 때만 이전 메시지를 붙이는 기존 로직 유지
            if (not single_turn) and getattr(session_history, "messages", None):
                # ChatMessageHistory는 보통 .messages가 list[BaseMessage] 이라 dict가 아닐 수 있음
                # 그대로 문자열화해서 맥락 힌트로만 전달
                prev_msgs_str = str([m.content for m in session_history.messages])
                response = chat_with_agent(user_input + "\n\nPrevious Messages: " + prev_msgs_str, agent_executor)
            else:
                response = chat_with_agent(user_input, agent_executor)

            # 현재 턴(질문/답)만 화면에 출력
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

            # 히스토리 저장 (단일 모드여도 이번 턴은 저장되어 다음 사이클 시작 전에 지워짐)
            try:
                # ChatMessageHistory 표준 메서드가 있으면 사용
                if hasattr(session_history, "add_user_message"):
                    session_history.add_user_message(user_input)
                    session_history.add_ai_message(response)
                else:
                    # 과거 방식 호환
                    session_history.add_message({"role": "user", "content": user_input})
                    session_history.add_message({"role": "assistant", "content": response})
            except Exception:
                pass

            print_messages()
            if single_turn:
                st.stop()  # 같은 실행 사이클에서 다른 출력이 섞이지 않도록
    else:
        st.warning("OpenAI API 키와 SerpAPI API 키를 입력하세요.")

if __name__ == "__main__":
    main()
