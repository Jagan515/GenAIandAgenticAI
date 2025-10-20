from langchain_google_genai import ChatGoogleGemini  # Switch to Google Gemini wrapper for Gemini compatibility
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os   

load_dotenv()

# Get your Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("Google API key not found! Check your .env file.")     
# Initialize ChatGoogleGemini pointing to Google Gemini
chat = ChatGoogleGemini(
    model="gemini-1.5-pro",  # Specify Gemini model
    temperature=0.7, #auto detects api key from env variable
    max_tokens=100
)
messages = [HumanMessage(content="Tell me a joke about programming.")]      
response = chat.invoke(messages)
print("Response from Gemini 1.5 Pro:")
print(response.content)