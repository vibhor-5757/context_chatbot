import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.summarize.chain import load_summarize_chain

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key= qdrant_api_key,
)

llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
        verbose=True
    )

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

url = str(input("Enter the URL of the youtube video you want summarized: "))

loader =YoutubeLoader.from_youtube_url(url)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ','], 
    chunk_size=1000, 
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

summarizer_chain = load_summarize_chain(llm, chain_type="map_reduce")
global_summary = summarizer_chain.run(docs)

tagged_docs = [
    {"content": doc.page_content, "metadata": {"type": "transcript_chunk"}}
    for doc in docs
]
tagged_docs.append({
    "content": global_summary,
    "metadata": {"type": "summary"}
})

qdrant_client.delete_collection(collection_name="yt_db")

qdrant_client.create_collection(
    collection_name="yt_db",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

qdrant = QdrantVectorStore(
    client=qdrant_client,
    collection_name="yt_db",
    embedding=embeddings
)

qdrant.add_documents(documents=tagged_docs)

# def get_retriever(query_type):
#     filter_type = {"Summary": ["summary"], 
#                    "Specific Question": ["transcript_chunk"]
#                 } 
#     filters = filter_type.get(query_type, None)

#     if filters:
#         return qdrant.as_retriever(
#             search_type="mmr",
#             search_kwargs={"k": 3, "filter": {"type": filters}}
#         )
#     else:
#         return qdrant.as_retriever(search_type="mmr", search_kwargs={"k": 3})

summary_retreiver = qdrant.as_retriever(search_type="mmr", search_kwargs= {"k":1,"filter" : {"type": "summary"}})
specific_retreiver = qdrant.as_retriever(search_type="mmr", search_kwargs= {"k":3, "filter": {"type": "transcript_chunk"}})

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Classify the intent of the following query: {query}\n"
         "Options: [Summary, Make a quiz, Specific Question]"),
    ]
)

classification_chain = classification_template | llm | StrOutputParser(output_key="classification")

include_history_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed. if the latest user question does not"
    "depend upon the chat history context, do NOT modify the user question."
)

include_history_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", include_history_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. keep the answer concise"
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# def get_retriever(llm, retriever, prompt):
#     return create_history_aware_retriever(
#         llm, retriever, prompt
#     )


include_history_chain = include_history_q_prompt | llm | StrOutputParser()
summary_chain = include_history_chain | summary_retreiver | qa_prompt | llm | StrOutputParser()
qa_chain = include_history_chain | specific_retreiver | qa_prompt | llm | StrOutputParser()

def generate_quiz_question(chunk, llm):
    prompt_template = (
        "You are a helpful assistant. Generate one quiz question and its answer "
        "based on the following text chunk:\n\n"
        "{chunk}\n\n"
        "Format the response as:\n"
        "Q: <Your question here>\n"
        "A: <Your answer here>"
    )
    prompt = prompt_template.format(chunk=chunk)
    return llm.run(prompt)

def generate_quiz(llm, retriever):
    retrieved_docs = retriever.get_relevant_documents("")
    questions_and_answers = []

    for doc in retrieved_docs:
        chunk_content = doc.page_content
        question_and_answer = generate_quiz_question(chunk_content, llm)
        questions_and_answers.append(question_and_answer)

    final_quiz = "\n\n".join(questions_and_answers)
    return final_quiz

quiz_chain = generate_quiz(llm, specific_retreiver)

branches = RunnableBranch(
    (
        lambda x: x["classification"] == "Summary",
        summary_chain
    ),
    (
        lambda x: x["classification"] == "Specific Question",
        qa_chain
    ),
    (
        lambda x: x["classification"] == "Make a quiz",
        quiz_chain
    )
)


rag_chain = classification_chain | branches
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

print("Start chatting with the AI! Type 'exit' to end the conversation.")
chat_history = []  
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    print(f"AI: {result['answer']}")
    chat_history.append(HumanMessage(content=query))
    chat_history.append(SystemMessage(content=result["answer"]))

