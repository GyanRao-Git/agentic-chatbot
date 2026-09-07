from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages

# checkpointing to store messages
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
from typing import TypedDict, Annotated
import os

load_dotenv()

"""
    intialise llm
    model gemini-3.6-flash
"""
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

#make state schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#make graph with state schema
graph = StateGraph(ChatState)


def chat_node(state: ChatState):
    message = state["messages"]

    res = llm.invoke(message)

    return {
        "messages": [res]
    }

graph.add_node("chat_node", chat_node)

#add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

checkpoint = MemorySaver()

chatbot = graph.compile(checkpointer= checkpoint)

initial_state: ChatState = {
    "messages" : [HumanMessage(content = "What is langgraph in 10 words")]
}

conv_id =1

while True:
    input_message = input("Type here: ")

    print("User: ", input_message , "\n")

    if input_message.strip().lower() in ["exit", "quit", "bye"]:
        break

    config = {
        'configurable':{
            "thread_id":conv_id
        }
    }



    initial_state["messages"] = [HumanMessage(content= input_message)]
    final_state = chatbot.invoke(initial_state, config= config)

    print("AI: ",final_state['messages'][-1].content[0]["text"])





