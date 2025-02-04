__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
import PyPDF2


def read_db(filepath: str, embeddings_name, k: int):
    """
    Function to read the vector database and assign at is retreiver

    Parameters:
        vdb_dir(str): the directory where the vector database is located
        embeddings(str): the embeddings name
        k(int): the number of results to return

    Returns:
        the retreiver
    """
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)
    vectordb = Chroma(persist_directory=filepath, embedding_function=embeddings)
    retreiver = vectordb.as_retriever(search_kwargs={"k": k})

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


def run_chain(rag_chain: RunnableWithMessageHistory, user_input: str, session_id: str):
    for chunk in rag_chain.stream(
        {"question": user_input}, config={"configurable": {"session_id": session_id}}
    ):
        if answer_chunk := chunk:
            yield answer_chunk


def extract_pdf_text(file_object):
    reader = PyPDF2.PdfReader(file_object)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return str(text)