import anthropic
from langchain_anthropic import ChatAnthropic  # Switch to Anthropic wrapper for Claude compatibility
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from openai import chat

load_dotenv()

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")  
if not anthropic_api_key:
    raise ValueError("Anthropic API key not found! Check your .env file.")

# Initialize ChatAnthropic pointing to Claude

chat = ChatAnthropic(
    model="claude-2",  # Specify Claude model
    temperature=0.7,
    max_tokens=100
)
messages = [HumanMessage(content="Tell me a joke about programming.")]      
response = chat.invoke(messages)
print("Response from Claude 2:")
print(response.content)