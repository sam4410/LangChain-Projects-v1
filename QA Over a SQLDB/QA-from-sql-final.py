import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
openai_api_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

sqlite_db_path = "data/street_tree_db.sqlite"
db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")

answer_prompt = PromptTemplate.from_template(
    """
    Given the user question,
    corresponding SQL query, and SQL result,
    answer the user question in a pretty format.
    
Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:
    """
)

write_query = create_sql_query_chain(llm, db)
# print(write_query)
execute_query = QuerySQLDataBaseTool(db=db)

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke({"question": "List the species of trees that are present in San Francisco"})
print(response)


