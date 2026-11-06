# from typing import List
from pydantic import BaseModel, Field, HttpUrl, ConfigDict

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
# from tavily import TavilyClient

# tavily = TavilyClient()

import time
from functools import wraps
def timer(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        end = time.time()
        print(f"for {fn.__name__}: time elapsed {round(end-start, 2)} sec.")
        return res
    return wrapper

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     """
#     print(f"Searching for {query}")
#     # return "Houston weather is sunny and cool"
#     return tavily.search(query=query)


class Source(BaseModel):
    """Schema for a source used by the agent"""
    model_config = ConfigDict(extra="forbid")
    url: HttpUrl = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(description="The agent's answer to the query")
    sources: list[Source] = Field(description="List of sources used to generate the answer; empty list if none")


llm = ChatOpenAI(model="gpt-5-nano")
#llm = ChatOllama(model="llama3.2:latest")

tools = [TavilySearch()]
# tools = [search]

# agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
                You are a helpful assistant. When a tool returns information, 
                  treat it as accurate and current, and answer the user using it.
                  """,
    response_format=AgentResponse
)

@timer
def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer in Oil and Gas industry in Houston TX on linkedin and list their details?"
                # content="what is the weather in Beijing today?"
            )
        }
    )
    print(result)



if __name__ == "__main__":
    main()
