from xml.dom.minidom import Document
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

Document=[
    "Hello mu friends",
    "My name is jagan pradhan",
    "I love to learn new things",
    "I am learning langchain library"
]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

result=embeddings.embed_documents(Document)  # Example usage

print("Embedding vector:", str(result))
