from langchain_openai import ChatOpenAI  # Switch to OpenAI wrapper for OpenRouter compatibility
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Get your OpenRouter API key
openrouter_api_key = os.getenv("OPENROUTER_API_KEY") # VARIABLE name remain same for all model of openrouter_api

if not openrouter_api_key:
    raise ValueError("OpenRouter API key not found! Check your .env file.")

# Initialize ChatOpenAI pointing to OpenRouter (for Anthropic model)
chat = ChatOpenAI(
    model="anthropic/claude-sonnet-4.5", #change the model name here for different model of openrouter
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1", # OpenRouter base URL same as before
    temperature=0.7,
    max_tokens=100  # Use max_tokens for OpenAI compat (replaces max_completion_tokens)
)

# Prepare messages
messages = [HumanMessage(content="Tell me a joke about programming.")]

# Get response
response = chat.invoke(messages)

print("Response from Claude Sonnet 4.5 via OpenRouter:")
print(response.content)