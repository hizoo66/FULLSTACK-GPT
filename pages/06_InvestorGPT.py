from typing import Type
from pydantic import BaseModel
from pydantic import Field
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.schema.runnable import RunnablePassthrough
from datetime import datetime

llm = ChatOpenAI(
    temperature=0.1,
)


class WikipediaSearchTool(BaseTool):

    name = "WikipediaSearchTool"
    description = """
    Use this tool to find the website for the given query.
    """

    class WikipediaSearchToolArgsSchema(BaseModel):
        query: str = Field(
            description="The query you will search for. Example query: Research about the XZ backdoor",
        )

    args_schema: Type[WikipediaSearchToolArgsSchema] = WikipediaSearchToolArgsSchema

    def _run(self, query):
        w = WikipediaAPIWrapper()
        return w.run(query)


class DuckDuckGoSearchTool(BaseTool):

    name = "DuckDuckGoTool"
    description = """
    Use this tool to find the website for the given query.
    """

    class DuckDuckGoSearchToolArgsSchema(BaseModel):
        query: str = Field(
            description="The query you will search for. Example query: Research about the XZ backdoor",
        )

    args_schema: Type[DuckDuckGoSearchToolArgsSchema] = DuckDuckGoSearchToolArgsSchema

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)


class LoadWebsiteTool(BaseTool):

    name = "LoadWebsiteTool"
    description = """
    Use this tool to load the website for the given url.
    """

    class LoadWebsiteToolArgsSchema(BaseModel):
        url: str = Field(
            description="The url you will load. Example url: https://en.wikipedia.org/wiki/Backdoor_(computing)",
        )

    args_schema: Type[LoadWebsiteToolArgsSchema] = LoadWebsiteToolArgsSchema

    def _run(self, url):
        loader = WebBaseLoader([url])
        docs = loader.load()
        return docs


class SaveToFileTool(BaseTool):
    name = "SaveToFileTool"
    description = """
    Use this tool to save the text to a file.
    """

    class SaveToFileToolArgsSchema(BaseModel):
        text: str = Field(
            description="The text you will save to a file.",
        )
        file_path: str = Field(
            description="Path of the file to save the text to.",
        )

    args_schema: Type[SaveToFileToolArgsSchema] = SaveToFileToolArgsSchema

    def _run(self, text, file_path):
        rearch_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{rearch_dt}_{file_path}", "w", encoding="utf-8") as f:
            f.write(text)
        return f"Text saved to {rearch_dt}_{file_path}"


def agent_invoke(input):

    agent = initialize_agent(
        llm=llm,
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        tools=[
            WikipediaSearchTool(),
            DuckDuckGoSearchTool(),
            LoadWebsiteTool(),
            SaveToFileTool(),
        ],
    )

    prompt = PromptTemplate.from_template(
        """    
        1. query 에 대해서 검색하고
        2. 검색 결과 목록에 website url 목록이 있으면, 각각의 website 내용을 text로 추출해서
        3. txt 파일로 저장해줘.
        
        query: {query}    
        """,
    )

    chain = {"query": RunnablePassthrough()} | prompt | agent
    chain.invoke(input)


query = "Research about the XZ backdoor"

agent_invoke(query)