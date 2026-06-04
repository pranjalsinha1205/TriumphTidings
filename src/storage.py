import pandas as pd
import os

OUTPUT_CSV = "triumph/data/conversations.csv"

def save_turn(session_id, turn_number, user_message, bot_message):
    convo = {
        "session_id": session_id,
        "turn_number": turn_number,
        "user_message": user_message,
        "bot_message": bot_message
    }
    df = pd.DataFrame([convo])
    df.to_csv(OUTPUT_CSV, mode="a", header=not os.path.exists(OUTPUT_CSV), index=False)
    return df

def get_conversation(session_id):
    try:
        df = pd.read_csv(OUTPUT_CSV)
        if session_id not in df["session_id"].unique():
            return 'No conversation for this session id'
        df1 = df[df.session_id == session_id]
        user = df1.user_message.tolist()
        bot = df1.bot_message.tolist()
        
        final_convo = ""
        for i in range(len(user)):
            final_convo = f"{final_convo} + \nUser: {user[i]}\n + Bot: {bot[i]}\n"
        return final_convo
    except Exception as e:
        return e