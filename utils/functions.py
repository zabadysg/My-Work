__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import PyPDF2
from utils.customllm import CustomLLM
from langchain_together import ChatTogether
from langsmith import traceable, trace
from uuid import uuid4

user_id=str(uuid4())
# def create_user_id():
#     return user_id

# control the temprature and the llm model type as input in the streamlit GUI

# llm = ChatTogether(model= "meta-llama/Llama-3.3-70B-Instruct-Turbo", temperature=0.0)

# # for streamlit GUI only
msgs = StreamlitChatMessageHistory(key="special_app_key")

# llm = CustomLLM(api_url="http://34.68.15.213:8000/chat_stream")


def read_db(filepath: str, embeddings_name):
    """
    Function to read the vector database and assign at is retreiver

    Parameters:
        vdb_dir(str): the directory where the vector database is located
        embeddings(str): the embeddings name

    Returns:
        the retreiver
    """
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)
    vectordb = Chroma(persist_directory=filepath, embedding_function=embeddings)
    retreiver = vectordb.as_retriever(search_kwargs={"k": 8})

    # take it as ret
    return retreiver





def read_system_prompt(filepath: str):
    """
    Function to read the system prompt

    Parameters:
        sys_prompt_dir(str): the directory where the system prompt is located

    Returns:
        The system prompt stored in variable
    """
    with open(filepath, "r") as file:
        prompt_content = file.read()

    context = "{context}"

    system_prompt = f'("""\n{prompt_content.strip()}\n"""\n"{context}")'

    return system_prompt


def create_rag_chain(sys_prompt_dir, vdb_dir, llm, embeddings_name):
    retriever = read_db(vdb_dir, embeddings_name)

    contextualize_q_system_prompt = """Given a chat history and the latest user question
    which might reference context in the chat history,
    formulate a response which can be understood and clear
    without the chat history. Do NOT answer the question,
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # add chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    sys_prompt = read_system_prompt(sys_prompt_dir)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def create_bot_for_selected_bot(
    embeddings, vdb_dir, sys_prompt_dir, msgs: StreamlitChatMessageHistory, llm
):
    """Create a bot for the selected configuration."""

    # llm = ChatTogether(model= "meta-llama/Llama-3.3-70B-Instruct-Turbo", temperature=0.0)
    # llm = CustomLLM(api_url="http://34.68.15.213:8000/chat_stream")

    rag_chain = create_rag_chain(sys_prompt_dir, vdb_dir, llm, embeddings)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
        max_tokens_limit=500,
        top_n=5,
    )
    return conversational_rag_chain


def _reduce_chunks(chunks: list):
    all_text = "".join([chunk for chunk in chunks])
    return all_text


# def feedback(feedback_text):
#     return feedback_text


@traceable(name="zabady", reduce_fn=_reduce_chunks, metadata={"user_id": user_id})
def bot_func(rag_chain, user_input, session_id):

    for chunk in rag_chain.stream(
        {"input": user_input}, config={"configurable": {"session_id": session_id}}
    ):
        if answer_chunk := chunk.get("answer"):
            yield answer_chunk


def extract_pdf_text(file_object):
    reader = PyPDF2.PdfReader(file_object)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return str(text)
