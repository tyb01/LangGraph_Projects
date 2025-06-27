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

class State(TypedDict):
    """
    Messages have the type "list" The "add_messages" function
    in the annotation defines how this state key should updates
    (in our case , it appends messages to the list ratHer than overwriting them)

    """

    messages:Annotated[list,add_messages]

graph_builder = StateGraph(State)



def chatbot(state:State):

    return {"messages":llm.invoke(state["messages"])}

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)

graph = graph_builder.compile()

# from IPython.display import Image, display
# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     pass


state = {"messages": []}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "q"]:
        print("Good Bye!")
        break

    state["messages"].append(("user", user_input))

    for event in graph.stream(state):
        # print(state["messages"])
        # print(state)
        # print(event.values())
        for value in event.values():
            state["messages"].append(("assistant", value["messages"].content))
            print("Assistant:", value["messages"].content)
