import os
import warnings
from langchain._api import LangChainDeprecationWarning
from langchain_core.messages import HumanMessage
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

chatbot = ChatOpenAI(model="gpt-3.5-turbo")

messagesToTheChatbot = [
    HumanMessage(content="My favorite color is blue."),
]

# chatbot with no memory
response = chatbot.invoke(messagesToTheChatbot)
print(response)

print(chatbot.invoke([
    HumanMessage(content="Do you remember my favorite color?")],
))

# adding memory to chatbot
from langchain import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.memory import FileChatMessageHistory

memory = ConversationBufferMemory(
    chat_memory = FileChatMessageHistory("message.json"),
    memory_key = "messages",
    return_messages=True
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chatbot,
    prompt=prompt,
    memory=memory
)

print(chain.invoke("hello!"))
print(chain.invoke("my name is Manoj Sharma"))
print(chain.invoke("do you remember my name?"))
print(chain.invoke("I want to gather some information about Football."))
print(chain.invoke("do you kow which game i want to gather information for?"))

# memory.clear()


