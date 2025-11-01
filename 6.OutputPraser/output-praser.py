from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from regex import template
load_dotenv()

llm =HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    task="text-generation",
    max_new_tokens=200
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


prompt1=template1.invoke({'topic':'black hole'})

result= model.invoke(prompt1)

prompt2=template2.invoke({'text':result.content})

result1=model.invoke(prompt2)
print(result1.content)