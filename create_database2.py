from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import json
from langchain.schema import Document
import shutil
import pickle

CHROMA_PATH = "chroma"
DATA_PATH = "data2"
QA_DICT_PATH = "qa_dict.pkl"


def main():
    generate_data_store()


def generate_data_store():

    create_qa_dict()
    documents = load_documents()
    # chunks = split_text(documents)
    save_to_chroma(documents)


def create_qa_dict():
    qa_dict = {}
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".json"):
            file_path = os.path.join(DATA_PATH, filename)
            with open(file_path, "r", encoding="utf8") as json_file:
                data = json.load(json_file)
                for item in data:
                    question = item.get("Question", "")
                    therapist_reply = item.get("TherapistReply", "")
                    qa_dict[question] = therapist_reply

    with open(QA_DICT_PATH, "wb") as f:
        pickle.dump(qa_dict, f)

    # return qa_dict


# def load_qa_dict():
#     if os.path.exists(QA_DICT_PATH):
#         with open(QA_DICT_PATH, "rb") as f:
#             return pickle.load(f)

#     return create_qa_dict()


def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".json"):
            file_path = os.path.join(DATA_PATH, filename)
            with open(file_path, "r", encoding="utf8") as json_file:
                data = json.load(json_file)
                for item in data:
                    question = item.get("Question", "")
                    # therapist_reply = item.get("TherapistReply", "")
                    # Concatenate question and therapist reply to form the document content
                    # content = f"Question: {question}\n\nTherapist Reply: {therapist_reply}"
                    # content = f"Question: {question}"
                    content = f"{question}"
                    document = Document(page_content=content, metadata={
                                        "question": question})
                    # document = Document(page_content=content)
                    documents.append(document)
    return documents


# def split_text(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=300,
#         chunk_overlap=100,
#         length_function=len,
#         add_start_index=True,
#     )
#     chunks = text_splitter.split_documents(documents)
#     print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

#     # document = chunks[10]
#     # print(document.page_content)
#     # print(document.metadata)

#     return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
