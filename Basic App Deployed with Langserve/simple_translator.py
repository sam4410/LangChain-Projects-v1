import os
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model="gpt-3.5-turbo")

system_prompt = "Translate the following text input into {language}:"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{text_input}")
    ]
)

chain = prompt | llm | StrOutputParser()

# deploying the application using Langserve
# Initialize the FastAPI app with metadata
app = FastAPI(
  title="simpleTranslator",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces that translates from Spanish to English",
)

# Add routes to the FastAPI app for the runnable chain
add_routes(
    app,
    chain,
    path="/chain",
)

# Run the server if the script is the main program
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

# to access langserve playground fro notebook
# from langserve import RemoteRunnable
#
# remote_chain = RemoteRunnable("http://localhost:8000/chain")
# remote_chain.invoke({"language":"Spanish",
#                      "text_input":"U.S. president Donald trump is proving to be a threat for entire world"})


