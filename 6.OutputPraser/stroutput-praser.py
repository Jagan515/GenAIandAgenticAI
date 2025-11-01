from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from regex import template
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm =HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

# 1 prompt - detail
template1=PromptTemplate(
    template='Write a detail report on {topic}',
    input_variables=['topic']
)
# 2 prompt - summary
template2=PromptTemplate(
    template='Write a 5 line summary on the following ./n{text}',
    input_variables=['text']
)

praser =StrOutputParser()

chain=template1 | model | praser | template2 | model | praser 

result=chain.invoke({'topic':'Black Hole'})

print(result)

