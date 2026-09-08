"""
    Interactive command-line entry point for the chatbot.
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from chatbot import ChatState, build_chatbot
from model_factory import create_chat_model


def main() -> None:
    load_dotenv()
    chatbot = build_chatbot(create_chat_model())
    conversation_id = "1"

    while True:
        user_input = input("Type here: ").strip()
        if user_input.lower() in {"exit", "quit", "bye"}:
            break

        state: ChatState = {"messages": [HumanMessage(content=user_input)]}
        config = {"configurable": {"thread_id": conversation_id}}
        final_state = chatbot.invoke(state, config=config)

        print(f"AI: {final_state['messages'][-1].text}")


if __name__ == "__main__":
    main()
