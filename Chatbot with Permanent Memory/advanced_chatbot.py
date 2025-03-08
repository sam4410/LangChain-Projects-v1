import os
import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# chatbot with no memory
chatbot = ChatOpenAI(model="gpt-3.5-turbo")

messagesToTheChatbot = [
    HumanMessage(content="My favorite color is red."),
]

response = chatbot.invoke(messagesToTheChatbot)
print(response)

# testing chatbot with no memory
print(chatbot.invoke([HumanMessage(content="Do you know what is my favorite color")]))

# Adding a permanent meory to our chatbot
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# below are 3 typical steps to build permanent memory
chatbotMemory = {}    # dict for storing memory

# input: session_id, output: chatbotMemory[session_id]
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]

chatbot_with_message_history = RunnableWithMessageHistory(
    chatbot,
    get_session_history
)

# invoking the chatbot giving a unique session id
session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite color is red.")],
    config=session1,
)
print(responseFromChatbot.content)

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="Do you know what is my favorite color?")],
    config=session1,
)
print(responseFromChatbot.content)

# switch session id to a different one
session2 = {"configurable": {"session_id": "002"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="Do you know what is my favorite color?")],
    config=session2,
)
print(responseFromChatbot.content)

# switch back to 'Session1' to validate if it still preserved the memory of user 1
responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="Do you know what is my favorite color?")],
    config=session1,
)
print(responseFromChatbot.content)

# Let's check if our chatbot remember the conversation from 'Session2'
responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="Hi, my name is Manoj Sharma")],
    config=session2,
)
print(responseFromChatbot.content)

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="Do you know my name?")],
    config=session2,
)
print(responseFromChatbot.content)

# Limiting the size of memory (managing the convesation history) avoid overflow for context window
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

def limited_memory_of_messages(messages, number_of_messages_to_keep=2):
    return messages[-number_of_messages_to_keep:]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

limitedMemoryChain = (
    RunnablePassthrough.assign(messages=lambda x: limited_memory_of_messages(x["messages"]))
    | prompt
    | chatbot
)

chatbot_with_limited_message_history = RunnableWithMessageHistory(
    limitedMemoryChain,
    get_session_history,
    input_messages_key="messages",
)

# let's add 2 more conversations to 'session1' conversation using chatbot with unlimited memory
responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite vehicles are Vespa scooters.")],
    config=session1,
)
print(responseFromChatbot.content)

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite city is San Francisco.")],
    config=session1,
)
print(responseFromChatbot.content)

# Now user 1 belong to session 1 has 4 messages in message history
# Now let's invoke chatbot with limited memory asking question from older messages in history (not last 2)
responseFromChatbot = chatbot_with_limited_message_history.invoke(
    {
        "messages": [HumanMessage(content="what is my favorite color?")],
    },
    config=session1,
)
print(responseFromChatbot.content)

# Now invoke chatbot with unlimited memory asking same question
responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="what is my favorite color?")],
    config=session1,
)
print(responseFromChatbot.content)

# print(chatbotMemory)