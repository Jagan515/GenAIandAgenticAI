from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# Load .env to get the Hugging Face token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found! Check your .env file.")

# Initialize LLM with a supported model (deployed for Inference API)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",  # Supported for chat_completion & text-generation
    huggingfacehub_api_token=hf_token,
    task="text-generation",  # Or "conversational" if available
    temperature=0.7,
    max_new_tokens=150
)

# Wrap with chat interface (handles templating automatically)
chat = ChatHuggingFace(llm=llm)

# Create a human message
messages = [HumanMessage(content="Tell me a funny programming joke!")]

# Get response
response = chat.invoke(messages)

print("🤖 SmolLM3-3B:", response.content)