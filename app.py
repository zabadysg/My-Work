from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
# Importing neccessary modules
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from datetime import datetime
import time


load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")


openai_client = wrap_openai(OpenAI())



########################################################




import streamlit as st
from langchain_openai import ChatOpenAI



# from langchain_together import ChatTogether
from uuid import uuid4
import pandas as pd
import base64
import os
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from utils.functions import *

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
msgs = StreamlitChatMessageHistory(key="langchain_messages")
BAAI = "BAAI/bge-base-en-v1.5"
L6 = "sentence-transformers/all-MiniLM-L6-v2"
names, sys_prompt_dirs, vdb_dirs = [], [], []
embeddings_list = []

for dir in os.listdir():
    if os.path.isdir(dir) and dir != "utils" and "." not in dir:
        names.append(dir)
        sys_prompt_dirs.append(f"{dir}/system_prompt.txt")
        vdb_dirs.append(f"{dir}/new_data_path/")
        embeddings_list.append(BAAI)

configurations = {
    "name": names,
    "embeddings_name": embeddings_list,
    "vdb_dir": vdb_dirs,
    "sys_prompt_dir": sys_prompt_dirs,
}
df = pd.DataFrame(configurations)
# save df to csv
df.to_csv("configurations.csv", index=False)


selected_bot = st.sidebar.selectbox("Choose a bot to interact with:", list(df["name"]))
st.write(f"{selected_bot} school Bot")

# TODO: MAKE DYNAMIC LOGO
accepted_extensions = [".png", ".jpg", ".jpeg"]

logo_path = None
# List all files in the directory
files = os.listdir(selected_bot)

# Find the first image file with accepted extension



st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <h1>{selected_bot}</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


selected_bot_config = df[df["name"] == selected_bot].iloc[0]
bot = create_bot_for_selected_bot(
    selected_bot_config["embeddings_name"],
    selected_bot_config["vdb_dir"],
    selected_bot_config["sys_prompt_dir"],
    msgs,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "bot_histories" not in st.session_state:
    st.session_state.bot_histories = {}

if "selected_bot" not in st.session_state:
    st.session_state.selected_bot = None

if selected_bot != st.session_state.selected_bot:
    if st.session_state.selected_bot is not None:  # Save the current bot's chat history
        st.session_state.bot_histories[st.session_state.selected_bot] = (
            st.session_state.get("chat_history", [])
        )

    # Load the selected bot's chat history or initialize it if not existing
    st.session_state.chat_history = st.session_state.bot_histories.get(selected_bot, [])
    st.session_state.selected_bot = selected_bot

uploaded_file = st.sidebar.file_uploader("Choose your .pdf file", type="pdf")
user_input = st.chat_input(placeholder="Your message")
user_input_2 = user_input

if uploaded_file is not None:
    text = extract_pdf_text(uploaded_file)
    if user_input:
        user_input += text
else:
    text=None

for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)

if user_input:
    doc={}
    with st.chat_message("user"):
        st.markdown(f"{user_input_2}")

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        session_id=str(uuid4())
        time_var=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()

        for response in bot_func(bot, user_input, session_id=session_id,puplic_doc=doc):
            full_response += response
            response_placeholder.markdown(f"{full_response}")
        
        end_time = time.time()
        latency = end_time - start_time

    @traceable(
    name="first try",
    run_type="retriever",
    metadata={"ls_provider": "SG_provider", "ls_model_name": "filtiration-Bot"}
    )
    def chat_model(user_input,run_id,time_var,latency):
        return doc,full_response

    st.session_state.chat_history.append(("You", f"{user_input_2}"))
    st.session_state.chat_history.append((selected_bot, f"{full_response}"))
