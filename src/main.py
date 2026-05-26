from rag import *
from storage import *
from diagnosis import *
import streamlit as st

st.title("Triumph Tidings")
st.caption("A mental health companion for college students")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = generate_session_id()

if "chain" not in st.session_state:
    st.session_state.chain = get_chain_with_history()

if "turn" not in st.session_state:
    st.session_state.turn = 0

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Talk to me...")

if question:
    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({"role": "user", "content": question})

    response = st.session_state.chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )

    if isinstance(response, dict):
        response = response.get("answer") or response.get("output") or str(response)

    st.session_state.turn += 1
    save_turn(st.session_state.session_id, st.session_state.turn, question, response)

    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    if st.button("End Session & Get Summary"):
        summary = diagnose(st.session_state.session_id)
        st.write(summary)