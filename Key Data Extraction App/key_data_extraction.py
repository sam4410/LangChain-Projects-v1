# objective: extract structured information from unstructured text
# for e.g. extracting first name, last name, country, phone no of users visiting website
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]
from langchain_openai import ChatOpenAI
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# define the information to be extracted using pydantic (library for data validation)
class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    lastname: Optional[str] = Field(
        default=None, description="The lastname of the person if known"
    )
    country: Optional[str] = Field(
        default=None, description="The country of the person if known"
    )

# Define the 'Extractor' chain
# Define a custom prompt to provide instructions and any additional context.
# 1) We can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)

chain = prompt | llm.with_structured_output(schema=Person)

user_review = """I absolutely love this product! It's been a game-changer for my daily routine. 
The quality is top-notch and the customer service is outstanding. I've recommended it to all my friends 
and family. - Sarah Johnson, USA"""

reponse = chain.invoke({"text": user_review})
print(reponse)

"""
This extraction capability is generative in nature which means that our model can perform a variety
of tasks beyond the expected. For e.g. it can infer gender of user based on their name, even when that
information is not provided explicitly. 
"""

# Extraction of list of entities rather then a single entity using 'nesting' models technique
class Persons(BaseModel):
    """Extracted data about people"""

    # Creates a model so we can extract multiple entities
    people: List[Person]

# chain
chain = prompt | llm.with_structured_output(schema=Persons)

reviews = """
Alice Johnson from Canada recently reviewed a book she loved. Meanwhile, Bob Smith from the USA 
shared his insights on the same book in a different review. Both reviews were very insightful.
"""

response = chain.invoke({"text": reviews})
print(response)



