from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load model & tokenizer locally 
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Fix for padding

# Detect device: MPS (Apple Silicon GPU) if available, else CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if device == "mps" else torch.float32,  # bfloat16 for MPS efficiency
    device_map="auto" if device != "cpu" else None,  # Auto-map on MPS; skip for CPU
    low_cpu_mem_usage=True,  # Reduces meta device offloading
    trust_remote_code=True  # For custom model code
)

# Create pipeline—omit 'device' param to let Accelerate handle placement
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
    # No 'device' here—fixed!
)

# Wrap in LangChain for chat interface (auto-applies chat template)
llm = HuggingFacePipeline(pipeline=pipe)
chat = ChatHuggingFace(llm=llm)

# Example messages
messages = [
    HumanMessage(content="Tell me a funny programming joke!")
]

# Get response
response = chat.invoke(messages)

print("TinyLlama (Local):", response.content)