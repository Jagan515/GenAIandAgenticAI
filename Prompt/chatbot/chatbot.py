from pyexpat import model
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
# System message to set the behavior of the assistant
system_message = SystemMessage(content="You are a helpful assistant that provides concise and accurate information.")

# Load .env to get the Hugging Face token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found! Check your .env file.")
# Initialize LLM with a supported model (deployed for Inference API)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    huggingfacehub_api_token=hf_token,
    task="text-generation",
    temperature=0.7,
    max_new_tokens=50
)
# Wrap with chat interface (handles templating automatically)
model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a helpful assistant that provides concise and accurate information.")
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break
    
    # Add user message first
    chat_history.append(HumanMessage(content=user_input))

    # Send to model
    response = model.invoke(chat_history)

    # Add model response to chat history
    chat_history.append(AIMessage(content=response.content))

    # Print reply
    print("SmolLM3-3B:", response.content)


print("Chat ended.")
print(chat_history)