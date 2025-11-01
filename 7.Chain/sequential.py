from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Generate a detail report in the {topic}",
    input_variables=['topic']
)

prompt2 =PromptTemplate(
    template="Generate a 5 line summary from these \n{text}",
    input_variables=["text"]
)
parser=StrOutputParser()
chain= prompt1 | model | parser | prompt2 | model | parser

response=chain.invoke({"topic":"Vector Database"})
print(response)
