from typing import Any, Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
import streamlit as st

with st.sidebar:
    st.title("OpenAI API Key")
    OPENAI_API_KEY = st.text_input("Write your API key", type="password")
    if OPENAI_API_KEY:
        openai_user_key = OPENAI_API_KEY
    else:
        st.write("please submit your API key")


llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=openai_user_key,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

memory = ConversationBufferMemory(memory_key="chat_history")


class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="The query to search for. Example: Stock Market Symbol for Apple."
    )


class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = "Search for stock market symbols using DuckDuckGo."
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = (
        StockMarketSymbolSearchToolArgsSchema
    )

    def _run(self, query: str):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)

    def _arun(self, query: str):

        raise NotImplementedError("This tool does not support async yet.")


agent = initialize_agent(
    tools=[StockMarketSymbolSearchTool()],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    handle_parsing_errors=True,
    verbose=True,
)

st.title("OpenAI Assistant Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.form(key="chat_form"):
    user_input = st.text_input("메시지를 입력하세요:")
    submit_button = st.form_submit_button("전송")

if submit_button and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        response = agent.invoke(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"에러 발생: {e}"}
        )


for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.write(f"**You**: {msg['content']}")
    else:
        st.write(f"**Assistant**: {msg['content']}")
