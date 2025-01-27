import streamlit as st

from uuid import uuid4
import uuid
import pandas as pd
import base64
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from utils.functions import *
from utils.customllm import *
from langsmith import traceable, trace,Client
from typing import Final

from langchain_together import ChatTogether
from streamlit_feedback import streamlit_feedback



import os
from dotenv import load_dotenv
# user_id = create_user_id() 
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
client = Client(api_key='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')


with st.sidebar:
    st.header("LLM Settings")
    
    # Dropdown for model type selection
    model_type = st.selectbox("Select LLM Model Type", ["ChatTogether","CustomLLM"])
    
    # Slider for temperature
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, key="temp_slider") 
    
    # Conditional inputs based on model type
    if model_type == "CustomLLM":
        api_url = st.text_input("Enter API URL", "http://34.68.15.213:8000/chat_stream", key="api_url")
        llm = CustomLLM(api_url=api_url)
    elif model_type == "ChatTogether":
        model_name = st.text_input("Enter Model Name", "meta-llama/Llama-3.3-70B-Instruct-Turbo", key="model_name")
        llm = ChatTogether(model=model_name, temperature=temperature,api_key='f3cd19b2691d15ac59cf87bda7b7e5b22e94e65a201479c9da5ee23547e8cb68')
    
    # Display the final llm variable
    st.subheader("LLM Variable")
    st.code(f"llm = {llm}")

msgs = StreamlitChatMessageHistory(key="langchain_messages")
BAAI = "BAAI/bge-base-en-v1.5"
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

selected_bot_config = df[df["name"] == selected_bot].iloc[0]
bot = create_bot_for_selected_bot(
    selected_bot_config["embeddings_name"],
    selected_bot_config["vdb_dir"],
    selected_bot_config["sys_prompt_dir"],
    msgs,
    llm
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

for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)

session_id=uuid.uuid5(uuid.NAMESPACE_DNS, str((len(st.session_state.chat_history)/2))+user_id)

if user_input:
    
    with st.chat_message("user"):
        st.markdown(f"{user_input_2}")

    with st.chat_message("assistant"):



        response_placeholder = st.empty()
        full_response = ""
        feedback=None

        for response in bot_func(bot, user_input, session_id=session_id,langsmith_extra={"run_id":session_id}): 
            full_response += response 
            response_placeholder.markdown(f"{full_response}")
    

    st.session_state.chat_history.append(("You", f"{user_input_2}"))
    st.session_state.chat_history.append((selected_bot, f"{full_response}"))  
    feedback = streamlit_feedback(feedback_type="thumbs",optional_text_label="[Optional] Please provide an explanation")


    if feedback:
        client.create_feedback(
            run_id=uuid.uuid5(uuid.NAMESPACE_DNS, str((len(st.session_state.chat_history)/2)-1)+user_id),
            key="User Feedback",
            score=1 if feedback['score']=='👍' else 0,
            comment=feedback['text'],
        )