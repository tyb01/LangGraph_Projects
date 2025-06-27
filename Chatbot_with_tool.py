import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "LANGRAPH-CHATBOT"

from langchain.chat_models import init_chat_model

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START , END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun , WikipediaQueryRun


arxiv_wrapper = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=400)

tool_arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)


wiki_wrapper = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=400)

tool_wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)


tools = [tool_wiki]

#Langgraph application
llm = init_chat_model("llama3-8b-8192", model_provider="groq")
llm = llm.bind(tools=tools)

class State(TypedDict):
    messages:Annotated[list,add_messages]

def chatbot(state:State):
    return {"messages": llm.invoke(state["messages"])}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,"chatbot")
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")


graph = graph_builder.compile()

state = {"messages": []}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "q"]:
        print("Good Bye!")
        break
    state["messages"].append(("user", user_input))
    for event in graph.stream(state):
        for value in event.values():
            state["messages"].append(("assistant", value["messages"].content))
            print("Assistant:", value["messages"].content)
