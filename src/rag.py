from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

API_SECRET = os.getenv("API_SECRET")

BOT_INFO = """
You are Triumph a mental health chatbot created especially for college students away from home.
You are to act as a companion, a bedfellow, and try to help a student tackle any sort of depression he might go through.
You are empathetic, lovable, and genuinely love everyone even with all their faults.
You listen to the user, but do not act like a philanthropist.
You gently question the user, if it is to help them overcome their problems.
You listen to their complaints, confessions, but never judge them.
You are to act while thinking you are in their place.
If the learner ever says anything about harming him/herself, you give out this helpline number: 1860-2662-345.
Try to calm them down, tell them jokes, tell them your own flaws, act as if the user isn't the only one confused.
You are like Sonya from Crime and Punishment.
You are to act like Jesus when the user is like Lazarus
"""

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma(
        collection_name="triumph_tidings",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore

def get_retriever():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever

def get_chat_prompt_temp():
    prompt = ChatPromptTemplate([
        ("system", BOT_INFO),
        ("system", "Use this {context} to answer. And while answering you shall keep history in context"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    return prompt

def format_docs(docs):
    return " ".join(doc.page_content for doc in docs)

def get_llm():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=API_SECRET
    )
    return llm

def create_and_get_chain():
    retriever = get_retriever()
    prompt = get_chat_prompt_temp()
    llm = get_llm()

    chain = RunnableParallel({
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": RunnablePassthrough(),
        "history": lambda x: x.get("history", [])
    }) | prompt | llm | StrOutputParser()

    return chain

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_chain_with_history(): 
    chain = create_and_get_chain()   
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    return chain_with_history

def generate_session_id():
    session_id = str(uuid.uuid4())[:8]
    return session_id
