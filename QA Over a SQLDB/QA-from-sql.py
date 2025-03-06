import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

llm =ChatOpenAI(model="gpt-3.5-turbo-0125")
sqlite_db_path = "data/street_tree_db.sqlite"
db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")

"""
Creating a simple chain that does the following:
* Convert the question into SQL query
* Execute the query
* Use te result to answer the question in natural language
"""
# chain = create_sql_query_chain(llm, db)
# response = chain.invoke({"question": "List the species of trees that are present in San Francisco?"})
# print(response)

# execute the query on db
# print(db.run(response))    #not a langchain way of getting results

# to check prompt in 'create_sql_query_chain' built by langchain
# chain.get_prompts()[0].pretty_print()

# Writing and Executing the SQL query prepared by 'create_sql_query_chain'
write_query = create_sql_query_chain(llm, db)   # step 1
execute_query = QuerySQLDataBaseTool(db=db)     # step 2

# create chain to bind the 2 steps
chain = write_query | execute_query
response = chain.invoke({"question": "List the species of trees that are present in San Francisco"})
print(response)
