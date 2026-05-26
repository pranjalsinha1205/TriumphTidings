from rag import *

def chat():
    session_id = generate_session_id()
    print(f"Session: {session_id} starts\n")
    chain_with_history = get_chain_with_history()
    while True:
        question = input("\nEnter your thoughts, to quit enter exit or quit: ")
        if question in ("quit", "exit"):
            print("\nThanks for talking with me!")
            break
        response = chain_with_history.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )
        print("Triumph:", response)

if __name__ == "__main__":
    chat()