import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI
openai_api_key = os.environ["OPENAI_API_KEY"]
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

file_path = "./data/Be_Good.pdf"
loader = PyPDFLoader(file_path=file_path)
docs = loader.load()
print(len(docs))
print(docs[0].page_content[:100])
print(docs[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=chunks, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

# We will use 2 pre-defined chains: create_stuff_document_chain and create_retrieval_chain
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", {input}),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

result = rag_chain.invoke({"input": "what is this article about?"})
print(result["answer"])
# print(result)   #will print input, context and answer keys of the RAG chain


