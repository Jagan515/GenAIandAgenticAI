from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Create embeddings
hf_emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Semantic splitter
splitter = SemanticChunker(
    embeddings=hf_emb,  # Note: 'embedding' -> 'embeddings' (plural, per API)
    breakpoint_threshold_type="percentile",  # Example: Use "standard_deviation" for fewer/larger chunks
    breakpoint_threshold_amount=90.0,  # Lower = larger chunks (default 95.0); range 0.0-100.0
    
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.
Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

docs = splitter.create_documents([sample])
print("Chunks:", len(docs))
for i, d in enumerate(docs):
    print(f"\n CHUNK {i} \n{d.page_content}")


