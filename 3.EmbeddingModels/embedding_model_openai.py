from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

result=embeddings.embed_query("My name is jagan pradhan")  # Example usage

print("Embedding vector:", str(result))
