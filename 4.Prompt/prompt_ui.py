from unittest import result
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
import os

# Load .env to get the Hugging Face token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found! Check your .env file.")

# Initialize LLM with a supported model (deployed for Inference API)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",  # Supported for chat_completion & text-generation
    huggingfacehub_api_token=hf_token,
    task="text-generation",  # Or "conversational" if available
    temperature=0.7,
    max_new_tokens=150
)

# Wrap with chat interface (handles templating automatically)
chat = ChatHuggingFace(llm=llm)
st.title("Chat with SmolLM3-3B")
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')

prompt= template.invoke()


# user_input =st.text_input("Enter your Prompt ")

if st.button("Generate Response"):
    chain =template | chat
    result=chain.invoke(
        {
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    }
    )

    st.text(result.content)   





