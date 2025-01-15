import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key= qdrant_api_key,
)

urls = ["https://www.apple.com/"]

loader = WebBaseLoader(urls)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ','], 
    chunk_size=1000, 
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
        verbose=True
    )

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

qdrant_client.delete_collection(collection_name="web_db")

qdrant_client.create_collection(
    collection_name="web_db",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

qdrant = QdrantVectorStore(
    client=qdrant_client,
    collection_name="web_db",
    embedding=embeddings
)

qdrant.add_documents(
    documents = docs
)

retriever = qdrant.as_retriever(search_type="mmr", search_kwargs={"k": 3})

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. keep the answer "
    "concise."
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

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

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

