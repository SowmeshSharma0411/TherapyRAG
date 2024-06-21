from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

import pickle

vectorstore = Chroma(persist_directory="chroma",
                     embedding_function=OpenAIEmbeddings(
                         openai_api_key=os.environ["OPENAI_API_KEY"]))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"], temperature=0)
# llm = ChatOpenAI(base_url='http://localhost:1234/v1',api_key="not-needed", temperature=0)

qa_dict = {}

with open('qa_dict.pkl', 'rb') as f:
    qa_dict = pickle.load(f)

    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)


def format_docs(docs):
    return [doc.page_content for doc in docs]


contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

# qa_system_prompt = """Adopt the role of an inquisitive therapist who seeks to understand the user's situation deeply. Ask probing questions to uncover underlying issues and provide insightful responses based on the context. Also suggest ways to overcome the situation as done in the context when it is needed. Use the following pieces of retrieved context to answer the question. Also ask follow up questions when applicable as seen in the context. If you dont know the answer or its not given in the retrieved context you can say that you dont have enough knowledge on the topic

# {context}"""
qa_system_prompt = """Adopt the role of an inquisitive therapist who seeks to understand the user's situation deeply. Ask probing questions to uncover underlying issues and provide insightful responses based on the context. Also suggest ways to overcome the situation as done in the context when it is needed. Please Use the following pieces of retrieved context: which is are Key-Value pairs of sample Question-Answers which are simialr to the user's query to help you understand on how to formulate your answer to the user's question. Also ask follow up questions when applicable as seen in the context. If you dont know the answer or its not given in the retrieved context you can say that you dont have enough knowledge on the topic

{similar_qna}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]


# def print_context(context):
#     print(len(context["context"]), "Context: \n", context["context"])
#     return context


# def print_actualContext(context):
#     new_context = {}
#     new_context['question'] = context['question']
#     new_context['chat_history'] = context['chat_history']

#     qna_retrieved = {}
#     for question in context['context']:
#         answer = qa_dict.get(question, "No answer found")
#         qna_retrieved[question] = answer
#         print(f"{question}\n")
#         print(f"{answer}\n")
#         print("-" * 50)

#     return context

def get_actualContext(context):
    new_context = {}
    new_context['question'] = context['question']
    new_context['chat_history'] = context['chat_history']

    qna_retrieved = {}
    for question in context['context']:
        answer = qa_dict.get(question, "No answer found")
        qna_retrieved[question] = answer

    new_context['similar_qna'] = qna_retrieved

    context = new_context

    return context


rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever | format_docs
    )
    | get_actualContext
    | qa_prompt
    | llm
    | StrOutputParser()
)

chat_history = []

while True:
    question = input("You: \n")
    if question == 'Stop':
        break
    ai_msg = rag_chain.invoke(
        {"question": question, "chat_history": chat_history})
    # print(ai_msg)
    print("Therapist:", ai_msg)
    chat_history.extend(
        [HumanMessage(content=question), AIMessage(content=ai_msg)])
