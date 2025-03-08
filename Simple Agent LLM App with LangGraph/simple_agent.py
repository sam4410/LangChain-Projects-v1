import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

"""
this basic agent is designed to autonomously choose if it want to use LLM or use searching internet
to answer a quetion. The purpose of this agent is to overcome the limitation of LLM models as
those are trained with all the internet data produced up to a certain point in time (cut off date).
So, LLM are not aware the recent events/information. For e.g. GPT-4o has been trained with all the
information on internet until Dec 2023
"""
online_search = TavilySearchResults(max_results=2)
# response = search.invoke("Who are the top stars of the 2024 Eurocup?")
# print(response)

tools = [online_search]

#to enable the model to do tool calling, we can use '.bind_tools' to give LLM knowledge of these tools
# llm_wth_tools = llm.bind_tools(online_search)
# instead of binding tools with LLM model, we will use agent to do that operation

# Create an agent
# agent_executor = create_react_agent(model=llm, tools=tools)

# run the agent
# response = agent_executor.invoke({"messages":[HumanMessage(content="where is the soccer Eurocup 2024 played?")]})
# print(response["messages"])

# running agent with .stream
# for chunk in agent_executor.stream(
#         {"messages":[HumanMessage(content="when and where will it be the 2025 ICC cup final match?")]}
# ):
#     print(chunk)
#     print("----")

# So far, agent does not have conversation ability as it has no memory
# Adding memory to agent
memory = MemorySaver()

agent_executor = create_react_agent(model=llm, tools=tools, checkpointer=memory)
config = {"configurable":{"thread_id": "001"}}

# run agent
for chunk in agent_executor.stream(
        {"messages": HumanMessage(content="Who won the 2024 soccer Eurocup?")}, config=config
):
    print(chunk)
    print("----")

# next question dependent on previous question as above
for chunk in agent_executor.stream(
        {"messages": HumanMessage(content="who were top stars of that winning team?")}, config=config
):
    print(chunk)
    print("----")

# now if we change the user_id or session_id in cofig
config = {"configurable": {"thread_id":"002"}}
for chunk in agent_executor.stream(
        {"messages": HumanMessage(content="About what soccer team we are talking about?")}, config=config
):
    print(chunk)
    print("----")
