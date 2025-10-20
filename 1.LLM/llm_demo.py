from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
llm = OpenAI(model="gpt-5-chat")

result=llm.invoke("Tell me a joke about programming.")
print(result)
# i am usig openrouter api key in .env file so error occur as it return string as ouptut not object
#The response is a structured object (usually a dictionary or a Python object in LangChain).
#to check if api key is loaded properly
# import os
# from dotenv import load_dotenv

# load_dotenv()
# key = os.getenv("OPENROUTER_API_KEY")

# if key:
#     print("Loaded successfully:", key)
# else:
#     print("Failed to load .env or variable missing!")

#No one Usees this method as its old method