import os
import streamlit as st
from uuid import uuid4
from dotenv import load_dotenv
from operator import itemgetter
import pandas as pd
# LangSmith and LangChain imports
from langsmith import Client, traceable
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Custom imports
from utils.customllm import *
from utils.functions import *

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# LangSmith Configuration
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "sg-program-explorer"

# Initialize LangSmith Client
langsmith_client = Client()

# LLM Initialization
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=4000)
embeddings_name = "BAAI/bge-base-en-v1.5"

# Initialize Streamlit Chat History
msgs = StreamlitChatMessageHistory(key="special_app_key")

names, sys_prompt_dirs, vdb_dirs = [], [], []
embeddings_list = []


for dir in os.listdir():
    if os.path.isdir(dir) and dir != "utils" and "." not in dir and dir != "additional data" and dir != "Visa":
        names.append(dir)
        sys_prompt_dirs.append(f"{dir}/system_prompt.txt")
        vdb_dirs.append(f"{dir}/Vdb/")
        embeddings_list.append(embeddings_name)


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
st.write(f"{selected_bot} Visa Bot")

selected_bot_config = df[df["name"] == selected_bot].iloc[0]


# System Prompt and Chain Setup
system_prompt = read_system_prompt(selected_bot_config["sys_prompt_dir"])
prompt = PromptTemplate.from_template(
    """
    #Previous Chat History:
    {chat_history}

    #Question: 
    {question} 
   """
    + system_prompt
    + """
    #Context: 
    {context} 

    #Answer:"""
)

# Vector Store Retriever
vdb_path = selected_bot_config["vdb_dir"]
retreiver = read_db(vdb_path, embeddings_name, k=5)

# Traceable RAG Chain
@traceable(run_type="chain")
def create_rag_chain(input_data: str, session_id: str):
    chain = (
        {
            "context": itemgetter("question") | retreiver,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    # Create a RAG chain that records conversations
    rag_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Function to retrieve session history
    input_messages_key="question",  # Key for the template variable that will contain the user's question
    history_messages_key="chat_history",  # Key for the history messages
)
    return run_chain(rag_with_history, input_data, session_id)

# Feedback Submission Function
@traceable(run_type="tool")
def submit_feedback(run_id, score, feedback_text, user_name):
    try:
        langsmith_client.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=score,
            comment=feedback_text,
            metadata={"user_name": user_name}
        )
        return True
    except Exception as e:
        st.error(f"Feedback submission failed: {e}")
        return False

# User Registration
def user_registration():
    st.sidebar.header("User Registration")
    user_name = st.sidebar.text_input("Enter your name")
    # email = st.sidebar.text_input("Enter your email")
    
    if st.sidebar.button("Register"):
        if user_name :
            st.session_state.user_name = user_name
            # st.session_state.user_email = email
            st.sidebar.success(f"Welcome, {user_name}!")
            return True
        else:
            st.sidebar.error("Please fill in all fields")
    return False

# Streamlit App
def main():
    st.title("SG-VisaBot")

    # User Registration Check
    if 'user_name' not in st.session_state:
        registration_success = user_registration()
        if not registration_success:
            return

    # Initialize session state
    if 'current_run_id' not in st.session_state:
        st.session_state.current_run_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # File Upload
    # uploaded_file = st.sidebar.file_uploader("Choose your .pdf file", type="pdf")
    user_input = st.chat_input(placeholder="Your message")

    # Render Chat History
    for sender, message in st.session_state.chat_history:
        with st.chat_message("user" if sender == "You" else "assistant"):
            st.markdown(message)

    # Process User Input
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            for response in create_rag_chain(
                    user_input, session_id=str(uuid4())
            ):
                full_response += response
                response_placeholder.markdown(f"{full_response}")
            
            # Invoke traceable RAG chain
            
            # Capture the run_id for feedback
            current_run = langsmith_client.list_runs(
                project_name="my-visa-bot", 
                execution_order=1, 
                limit=1
            )
            
            if current_run:
                st.session_state.current_run_id = current_run[0].id

        # Update chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Assistant", full_response))

        # Feedback Section
        if st.session_state.current_run_id:
            st.markdown("### Provide Feedback")
            feedback_col1, feedback_col2 = st.columns(2)
            
            with feedback_col1:
                feedback_score = st.select_slider(
                    "Rate the response:", 
                    options=[1, 2, 3, 4, 5],
                    value=3
                )
            
            with feedback_col2:
                feedback_text = st.text_input("Additional comments (optional)")
            
            if st.button("Submit Feedback"):
                if submit_feedback(
                    st.session_state.current_run_id, 
                    feedback_score, 
                    feedback_text,
                    st.session_state.user_name
                ):
                    st.success("Feedback submitted successfully!")

# Run the Streamlit app
if __name__ == "__main__":
    main()