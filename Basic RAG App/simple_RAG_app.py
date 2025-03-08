import os
import bs4
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
openai_api_key = os.environ['OPENAI_API_KEY']
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

loader = TextLoader("./data/be-good.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# pre-defined prompt template from langchain-hub
# prompt = hub.pull("rlm/rag-prompt")   # not compatible with python 3.11

# to keep using python 3.11.4 we will paste the prompt from hub
prompt  = ChatPromptTemplate(input_variables=['context', 'question'],
                             metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt',
                                       'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'},
                             messages=[HumanMessagePromptTemplate(
                                 prompt=PromptTemplate(input_variables=['context', 'question'],
                                                       template="""You are an assistant for question-answering tasks. 
                                                       Use the following pieces of retrieved context to answer the question. 
                                                       If you don't know the answer, just say that you don't know. 
                                                       Use three sentences maximum and keep the answer concise."""
                                                                "\nQuestion: {question} "
                                                                "\nContext: {context} "
                                                                "\nAnswer:"))])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("what is this document about?")
print(response)

