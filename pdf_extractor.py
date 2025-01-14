from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient, models
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

from dotenv import load_dotenv
import os

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

loader = PyPDFLoader("./data/[1]Harry Potter and the Philosopher-s Stone.pdf")

pages = []
for doc in loader.lazy_load():
    pages.append(doc)

# pages = pages[:120]


qdrant_client.delete_collection(collection_name="pdfs_db")

qdrant_client.create_collection(
    collection_name="pdfs_db",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)
# Append to an existing collection
qdrant = QdrantVectorStore(
    client=qdrant_client,
    collection_name="pdfs_db",
    embedding=embeddings
)
try:
    batch_size = 10
    for i in range(0, len(pages), batch_size):
        batch = pages[i:i + batch_size]
        qdrant.add_texts(
            texts=[doc.page_content for doc in batch],
            metadatas=[doc.metadata for doc in batch]
        )
except Exception as e:
    print(f"Error: {e}")
# results = qdrant.similarity_search(
#     "is harry potter a wizard?", k=3
# )
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")

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




