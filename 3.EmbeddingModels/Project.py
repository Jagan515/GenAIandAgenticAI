from langchain_huggingface import HuggingFaceEmbeddings  # For creating embeddings
from sklearn.metrics.pairwise import cosine_similarity  # For measuring similarity
import numpy as np  

class SimpleEmbeddingQA:
    def __init__(self, statements, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        
        self.statements = statements  # Our "documents" as a list
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)  # Load embedding model
        self.statement_embeddings = []  # Will hold embeddings for each statement
        self.embed_statements()  # Embed them right away
        print(f"Embedded {len(self.statements)} statements.")

    def embed_statements(self):
        
        self.statement_embeddings = self.embeddings.embed_documents(self.statements)

    def query(self, question, top_k=3):
    
        # Step 1: Embed the question
        question_embedding = self.embeddings.embed_query(question)
        
        # Step 2: Calculate cosine similarity between question and all statements
        similarities = cosine_similarity([question_embedding], self.statement_embeddings)[0]
        
        # Step 3: Get the top-k most similar statements (sorted by score)
        top_indices = np.argsort(similarities)[::-1][:top_k]  # Highest scores first
        
        # Step 4: Build the answer by combining top statements
        answer = f"Based on the statements, here are the top {top_k} relevant ones:\n\n"
        for idx in top_indices:
            statement = self.statements[idx]
            score = similarities[idx]
            answer += f"Similarity: {score:.4f}\n  {statement}\n\n"
        
        return answer


if __name__ == "__main__":
    # Step 1: Define your list of statements (like mini-documents)
    statements = [
        "Python is a programming language created by Guido van Rossum in 1991.",
        "The capital of France is Paris, known for the Eiffel Tower.",
        "Machine learning is a subset of artificial intelligence.",
        "The sky is blue due to Rayleigh scattering.",
        "Hugging Face provides open-source AI models and tools."
    ]
    
    # Step 2: Create the QA system
    qa = SimpleEmbeddingQA(statements)
    
    # Step 3: Get question from user (interactive!)
    question = input("Enter your question: ").strip()  # Takes input from user
    if not question:
        question = "What is Python?"  # Default if empty
    
    # Step 4: Get and print the answer
    answer = qa.query(question, top_k=2)  # Get top 2 matches
    
    print(f"\nQuestion: {question}\n")
    print(answer)