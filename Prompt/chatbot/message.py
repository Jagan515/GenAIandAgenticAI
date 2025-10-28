from re import A
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found! Check your .env file.")
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    huggingfacehub_api_token=hf_token,
    task="text-generation",
    temperature=0.7,
    max_new_tokens=50
)
model = ChatHuggingFace(llm=llm)

# System message to set the behavior of the assistant
#system_message = SystemMessage(content="You are a helpful assistant that provides concise and accurate information.")
message=[
    SystemMessage(content="You are a helpful assistant that provides concise and accurate information."),
    HumanMessage(content="What is Langchain and ai agents?"),
]
respone=model.invoke(message)
message.append(AIMessage(content=respone.content))
print("SmolLM3-3B:", respone.content)
