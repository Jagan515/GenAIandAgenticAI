from langchain_huggingface import HuggingFaceEmbeddings
from numpy import vecdot

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# test="My name is jagan pradhan"
Document=[
    "Hello mu friends",
    "My name is jagan pradhan",
    "I love to learn new things",
    "I am learning langchain library"
]
# vector_stores=embedding_model.embed_query(test)  # Example usage
vecdtor_stores=embedding_model.embed_documents(Document)  # Example usage

# print("Embedding vector:", str(vector_stores))
print("Embedding vector:", str(vecdtor_stores))