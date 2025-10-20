from langchain_openai import ChatOpenAI  
from langchain_core.messages import HumanMessage  # Updated import for HumanMessage
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Get your OpenRouter API key
openrouter_api_key = os.getenv("OPENAI_API_KEY")

if not openrouter_api_key:
    raise ValueError("OpenAI API key not found! Check your .env file.")

# Initialize ChatOpenAI pointing to OpenRouter
chat = ChatOpenAI(
    model="chatgpt-4o-latest",  # Note: Use 'model' instead of 'model_name' in newer versions
      # Note: Use 'base_url' instead of 'openai_api_base'
    temperature=0.7,
    max_completion_tokens=100
)

# Prepare messages
messages = [HumanMessage(content="Tell me a joke about programming.")]

# Get response
response = chat.invoke(messages)  # Use .invoke() for the list of messages (replaces direct call)

print("Response from GPT-4o via OpenRouter:")
print(response.content)