import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    political_tendency: str = Field(
        description="The political tendency of the user"
    )
    language: str = Field(description="The language the text is written in")


# LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125").with_structured_output(Classification)

# chain
chain = tagging_prompt | llm

trump_follower = """I'm confident that President Trump's leadership and track record will once again 
resonate with Americans. His strong stance on economic growth and national security is exactly what 
our country needs at this pivotal moment. We need to bring back the proven leadership that can make 
America great again!"""
biden_follower = """I believe President Biden's compassionate and steady approach is vital for our 
nation right now. His commitment to healthcare reform, climate change, and restoring our international 
alliances is crucial. It's time to continue the progress and ensure a future that benefits 
all Americans."""

response1 = chain.invoke({"input": trump_follower})
response2 = chain.invoke({"input": biden_follower})
print(response1)
print(response2)

# improving the reponse by limiting the classes in classification for chat model to choose from
class Classification_with_limited_classes(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    political_tendency: str = Field(
        ...,
        description="The political tendency of the user",
        enum=["conservative", "liberal", "independent"],
    )
    language: str = Field(
        ..., enum=["spanish", "english"]
    )

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0).with_structured_output(
    Classification_with_limited_classes)

chain = tagging_prompt | llm

biden_follower2 = """
Creo que el enfoque compasivo y constante del presidente Biden es vital para nuestra 
nación en este momento. Su compromiso con la reforma de la atención médica, el cambio climático y la restauración de nuestras 
alianzas internacionales es crucial. Es hora de continuar el progreso y garantizar un futuro que beneficie a todos los 
estadounidenses.
"""

response3 = chain.invoke({"input": trump_follower})
response4 = chain.invoke({"input": biden_follower2})
print(response3)
print(response4)