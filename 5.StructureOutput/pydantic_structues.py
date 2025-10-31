from dotenv import load_dotenv
from typing import List, Optional, Literal

from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# LLM setup (same as yours) 
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B", # Does not support pydantic ,model like OpenAi,Gemini support
    task="text-generation",
    max_new_tokens=20,
)

model = ChatHuggingFace(llm=llm)

#Pydantic model for structured output 
class Review(BaseModel):
    key: List[str] = Field(..., description="Write the key themes found in the review")
    summary: str = Field(..., description="A brief summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        ..., description="Sentiment: positive, negative, or neutral"
    )
    pros: Optional[List[str]] = Field(None, description="List the pros (if any)")
    cons: Optional[List[str]] = Field(None, description="List the cons (if any)")

# Ask the model to output structured data validated against Review 
structured = model.with_structured_output(Review)

text = """
Not only that, but Black Friday 2025 is fast approaching, and once again I expect to see loads of brilliant laptop deals...
(keep the rest of your long review text here)
"""

raw_response = structured.invoke(text)

# raw_response could already be a dict that matches the model. Validate/normalize via Pydantic:
# For pydantic v2 use model_validate; for v1 use parse_obj
try:
    # pydantic v2
    validated: Review = Review.model_validate(raw_response)
except AttributeError:
    # fallback for pydantic v1
    validated: Review = Review.parse_obj(raw_response)

# Now you can access typed attributes:
print("Summary:", validated.summary)
print("Sentiment:", validated.sentiment)
print("Key themes:", validated.key)
print("Pros:", validated.pros)
print("Cons:", validated.cons)

# If you want JSON:
print("As JSON:", validated.model_dump_json())   # v2
# (if using v1: validated.json())
