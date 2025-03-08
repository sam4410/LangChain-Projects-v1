import os
import bs4
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
openai_api_key = os.environ["OPENAI_API_KEY"]
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

loader = TextLoader("./data/be-good.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

"""
Process to follow:
1. Create a basic RAG without memory
2. Create a ChatPromptTemplate be able to contextualize the input question
3. Create a retriever ware of memory
4. Create a basic conversational RAG
5. Create an advanced conversational RAG with persistence and session memories 
"""
vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

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
        ("human", "{input}")
    ]
)

# 'create_stuff_documents_chain' will build as QA chain: chain asking question to LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# 'create_retrieval_chain' along with QA chain defined above build the RAG chain: a chain able to
# ask questions to retriever and then format the response with the LLM
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "what is this article about?"})
print(response["answer"])

# asking a second question related to previous question
response1 = rag_chain.invoke({"input": "what was my previous question about?"})
print(response1["answer"])    # throw hallucinated response

# step 2: Create a ChatPromptTemplate able to contextualize inputs
# (take user inout and rephrase it in context of user's previous conversation)
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# provide contextualized system prompt, MessagePlaceHolder (passes list of messages included in chat history)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Step 3: Create retriever aware of memory
# this retriever is going to take contextuaize input from step 2
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Step 4: Create a basic conversational RAG
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),    # using the basic system prompt built in step 1
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# trying out this basic conversational RAG
chat_history = []

question = "what is this article about?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})

chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"])
    ]
)

second_question = "what waas my previous question about?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
print(ai_msg_2["answer"])

# Step 5: Advanced conversational RAG wih persistence and session memories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# trying advanced conversational rag chain
response = conversational_rag_chain.invoke(
    {"input": "what is this article about?"},
    config={
        "configurable": {"session_id": "001"}
    }
)["answer"]
print(response)

next_response = conversational_rag_chain.invoke(
    {"input": "what was my previous question about?"},
    config={
        "configurable": {"session_id": "001"}
    }
)["answer"]
print(next_response)






