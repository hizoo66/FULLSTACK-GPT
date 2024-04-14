import os
import json
from datetime import datetime
import streamlit as st
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
from openai import OpenAI
import openai as client

def main():
    st.set_page_config(
        page_title="Assistant",
        page_icon="❌",
    )

    st.title("D33 | OpenAI Assistants (졸업 과제)")

    with st.sidebar:
        if "api_key" not in st.session_state:
            st.session_state["api_key"] = ""

        api_key_input = st.empty()

        def reset_api_key():
            st.session_state["api_key"] = ""

        api_key = api_key_input.text_input(
            value=st.session_state["api_key"],
            key="api_key_input",
        )

        if api_key != st.session_state["api_key"]:
            st.session_state["api_key"] = api_key
            st.rerun()
    
        github_url = st.text("https://github.com/hizoo66/FULLSTACK-GPT/blob/main/pages/07_AssistantsAPI.py")
        app_url = st.text("https://fullstack-gpt-8ewig3twyrqpwf5kis6bet.streamlit.app/AssistantsAPI")
        maker = st.text("made by Hizoo")
     

        log_box = st.expander(":blue[Assistant API Log]", expanded=True)

        if "logs" not in st.session_state:
            st.session_state["logs"] = []

        if log_box.button("Clear Log"):
            st.session_state["logs"] = []

        def paint_logs():
            for log in st.session_state["logs"]:
                log_box.write(log)

        def add_log(*args, seperator=" | "):
            log_message = seperator.join(str(arg) for arg in args)
            log = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')[:-3]}{seperator}{log_message}"
            st.session_state["logs"].append(log)
            log_box.write(log)

        paint_logs()

    if not api_key:
        st.warning("API키를 입력해주세요!")
        st.stop()

    client = OpenAI(api_key=api_key)

    def get_websites_by_wikipedia_search(inputs):
        w = WikipediaAPIWrapper()
        query = inputs["query"]
        return w.run(query)

    def get_websites_by_duckduckgo_search(inputs):
        ddg = DuckDuckGoSearchAPIWrapper()
        query = inputs["query"]
        return ddg.run(query)

    def get_document_text(inputs):
        url = inputs["url"]
        loader = WebBaseLoader([url])
        docs = loader.load()
        return docs[0].page_content

    functions_map = {
        "get_websites_by_wikipedia_search": get_websites_by_wikipedia_search,
        "get_websites_by_duckduckgo_search": get_websites_by_duckduckgo_search,
        "get_document_text": get_document_text,
    }

    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_websites_by_wikipedia_search",
                "description": "Use this tool to find the websites for the given query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query you will search for. Example query: Research about the XZ backdoor",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_websites_by_duckduckgo_search",
                "description": "Use this tool to find the websites for the given query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query you will search for. Example query: Research about the XZ backdoor",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_document_text",
                "description": "Use this tool to load the website for the given url.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The url you will load. Example url: https://en.wikipedia.org/wiki/Backdoor_(computing)",
                        }
                    },
                    "required": ["url"],
                },
            },
        },
    ]

    assistant = client.beta.assistants.create(
        name="Research Assistant",
        instructions="If there is a website url list in the search results list, extract each website content as a text. ",
        model="gpt-3.5-turbo",
        tools=functions,
    )
    assistant