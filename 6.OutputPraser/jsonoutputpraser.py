from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

llm =HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()
template1=PromptTemplate(
    template="Give me the name,age and city of the fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()} #its passed before the run time

)
# prompt=template1.format()

# result=model.invoke(prompt)

# final_result=parser.parse(result.content) 

chain=template1 | model | parser

result=chain.invoke({}) # passing empty dictional as the input_variable is empty
#Biggest fault of json putput parserv is its does not apply schema 
print(result)