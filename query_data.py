import argparse
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a therapist. Answer the questions based on the provided data in the context.

{context}

---

Answer the question based on the above context: {question}
Give a longer response as similar to these 3 responses by combining or paraphrasing these responses such that it sounds as human as possible. Answer in 2nd person throughout the response. Suggest possible solutions that might help the user based on the suggestions given in the 3 responses.  
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(
        openai_api_key=os.environ["OPENAI_API_KEY"])
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"])
    response_text = model.predict(prompt)
    formatted_response = f"Response: {response_text}\n"
    print(formatted_response)

    pass


if __name__ == "__main__":
    main()
