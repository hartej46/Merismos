from .conversation_store import save_conversation
from .message_store import save_message
from .memory_store import save_memory


def start_chat(response):       #from ai
    conversation_id = save_conversation("merismos-chat")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() == "\exit":
            break
        user_message_id = save_message(
            conversation_id=conversation_id,
            role="user",
            content=user_text,
        )

        save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response,
        )
        print(f"AI: {response}")
        if user_text.lower().startswith("remember:"):
            save_memory(                                                    #when prod , a new model decides what to do
                content=user_text.removeprefix("remember:").strip(),
                memory_type="user_saved",
                importance=1.0,
                confidence=1.0,
                source_message_id=user_message_id,
            )

if __name__ == "__main__":
    start_chat("Demo char")