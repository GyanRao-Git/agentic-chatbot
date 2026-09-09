"""
    Interactive command-line entry point for the chatbot.
"""
import os
import psycopg

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Persistent state using PostgreSQL checkpointing
from langgraph.checkpoint.postgres import PostgresSaver

from chatbot import ChatState, build_chatbot
from model_factory import create_chat_model
from conversations_repository import create_conversation


def main() -> None:
    load_dotenv()
    database_url = os.environ["DATABASE_URL"]
    with (
        PostgresSaver.from_conn_string(database_url) as checkpointer,
        psycopg.connect(database_url, autocommit=True) as database_connection
    ):
        checkpointer.setup()
        model = create_chat_model(provider = "gemini")
        chatbot = build_chatbot(
            model=model,
            checkpointer=checkpointer
        )
        conversation_id = input(
            "Conversation ID (leave blank for new):").strip()

        if not conversation_id:
            conversation_id = str(create_conversation(database_connection))
            print("New conversation with id: ", conversation_id, "\n")
        else:
            print("Resuming conversation with id: ", conversation_id, "\n")

        while True:
            user_input = input("Type here: ").strip()
            if user_input.lower() in {"exit", "quit", "bye"}:
                break

            # do not append, checkpointer will append, only send new message to avoid duplicates
            state: ChatState = {
                "messages": [HumanMessage(content=user_input)]
            }
            config = {
                "configurable": {
                    "thread_id": conversation_id
                }
            }
            final_state = chatbot.invoke(state, config=config)

            print(f"AI: {final_state['messages'][-1].text}")


if __name__ == "__main__":
    main()
