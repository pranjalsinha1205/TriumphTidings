from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
import pandas as pd
from dotenv import load_dotenv
from storage import *

load_dotenv()

API_SECRET = os.getenv("API_SECRET")

OUTPUT_CSV = "../data/summaries.csv"

PROMPT = """
You are an emotional conversation summarizer. 
You are given a conversation between a user and the bot.
You need to make a diagnosis based on that conversation.
The diagnosis includes:-
i) summary
ii) user mental state
iii) user emotional stability
iv) main point of conversation
v) overall user mental health assessment

You need to write a summary with all of these.

You have to make use of this conversation
{context}
"""

def diagnose(session_id):
    conversation = get_conversation(session_id)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=API_SECRET
    )
    prompt = ChatPromptTemplate([
        ("human", PROMPT)
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": conversation})

    summ = {
        "session_id": session_id,
        "summary": response
    }

    df = pd.DataFrame([summ])
    df.to_csv(OUTPUT_CSV, mode="a", header=not os.path.exists(OUTPUT_CSV), index=False)
    return response