import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
import re

# Simple config
SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi"}
# Map to YouTube lang codes
LANG_MAP = {"en": "en", "hi": "hi"}

# Load the small LLM (cached so it loads once)
@st.cache_resource
def load_llm():
    """
    Load a small text generation model via LangChain's HuggingFacePipeline.
    Runs on CPU, no GPU needed. This is lightweight and fast for short transcripts.
    """
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
    return HuggingFacePipeline(pipeline=pipe)

# Helper to extract video ID from URL
def get_video_id(url):
    """
    Pull out the 11-character video ID from a YouTube URL.
    Examples: https://www.youtube.com/watch?v=ABC123 or https://youtu.be/ABC123
    """
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Get transcript from YouTube (updated for new API)
def get_transcript(video_id, lang="en"):
    """
    Fetch the video transcript using YouTube's API.
    Tries manual captions first, then auto-generated.
    Returns the full text joined together.
    """
    ytt_api = YouTubeTranscriptApi() # Instantiate as per new API
    try:
        # List available transcripts (new method: .list(video_id))
        transcript_list = ytt_api.list(video_id)
       
        # Try preferred language
        target_lang = LANG_MAP.get(lang, "en")
        for auto_gen in [False, True]: # Manual first, then auto
            try:
                if auto_gen:
                    transcript = transcript_list.find_generated_transcript([target_lang])
                else:
                    transcript = transcript_list.find_manually_created_transcript([target_lang])
               
                # Fetch and join text chunks (use .text attribute)
                data = transcript.fetch()
                return " ".join(chunk.text for chunk in data)
            except NoTranscriptFound:
                continue # Try next option
       
        # Fallback to English
        for auto_gen in [False, True]:
            try:
                if auto_gen:
                    transcript = transcript_list.find_generated_transcript(["en"])
                else:
                    transcript = transcript_list.find_manually_created_transcript(["en"])
                data = transcript.fetch()
                return " ".join(chunk.text for chunk in data)
            except NoTranscriptFound:
                continue
       
        raise RuntimeError("No transcripts found.")
   
    except TranscriptsDisabled:
        raise RuntimeError("This video has no captions available.")

# Process video (main function)
def process_video(url, lang="en"):
    """
    Step 1: Extract video ID.
    Step 2: Get transcript.
    Returns the full transcript text.
    """
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL. Check the link.")
   
    st.info("Fetching transcript...")
    transcript = get_transcript(video_id, lang)
    return transcript

# Ask the LLM a question using sequential chaining (Runnable concepts)
def ask_question(transcript, question, llm):
    """
    Sequential chaining: First condense the transcript (Runnable chain), 
    then answer the question based on the condensed version.
    Uses Runnable | operator for chaining.
    Prompts designed to stick to transcript only.
    """
    # Step 1: Condense chain (Runnable)
    condense_template = """Condense the following transcript into < key sentences. Use only the provided transcript.
    Transcript: {transcript}
    Condensed:"""
    condense_prompt = PromptTemplate.from_template(condense_template)
    condense_chain = condense_prompt | llm | StrOutputParser()
    
    # Step 2: Answer chain (Runnable) - Strict to transcript
    answer_template = """Using ONLY the condensed transcript below, answer the question. 
    Do not use any external knowledge or your knowledge base. 
    If the answer is not directly supported by the condensed transcript, respond exactly with 'I don't know.'
    Condensed Transcript: {condensed}
    Question: {question}
    Answer:"""
    answer_prompt = PromptTemplate.from_template(answer_template)
    answer_chain = answer_prompt | llm | StrOutputParser()
    
    # Sequential execution
    with st.spinner("Condensing transcript..."):
        condensed = condense_chain.invoke({"transcript": transcript})
    with st.spinner("Generating answer..."):
        answer = answer_chain.invoke({"condensed": condensed, "question": question})
    
    return answer, condensed  # Return both for potential display

# Generate insights using parallel chaining (Runnable concepts)
def generate_insights(transcript, llm):
    """
    Parallel chaining: Run summary and entity extraction in parallel using RunnableParallel.
    Prompts designed to stick to transcript only.
    """
    # Summary chain
    summary_template = """Summarize the video transcript in  short bullet points.
    Keep each point under 400 words. Use only the provided transcript.
    Transcript: {transcript}
    Summary:"""
    summary_prompt = PromptTemplate.from_template(summary_template)
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    # Entities chain
    entities_template = """Extract the main entities (people, places, organizations, concepts) from the transcript ONLY.
    List them as a comma-separated list. Do not add any external knowledge.
    Transcript: {transcript}
    Entities:"""
    entities_prompt = PromptTemplate.from_template(entities_template)
    entities_chain = entities_prompt | llm | StrOutputParser()
    
    # Parallel Runnable
    parallel_chain = RunnableParallel(
        summary=summary_chain,
        entities=entities_chain
    )
    
    insights = parallel_chain.invoke({"transcript": transcript})
    return insights

# Generate a simple summary using Runnable chain (no LLMChain)
def generate_summary(transcript, llm):
    """
    Runnable chain for summary (replacing legacy LLMChain).
    Prompt designed to stick to transcript only.
    """
    template = """Summarize the video transcript in <500 words. Use only the provided transcript.
    Transcript: {transcript}
    Summary:
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    # Invoke chain
    summary = chain.invoke({"transcript": transcript})
    
    return summary

# --- Simple Streamlit UI ---
def main():
    # Set up the page
    st.set_page_config(page_title="Simple YouTube Q&A", page_icon="🎥", layout="wide")
    st.title("🎥 Simple YouTube Q&A App")
    st.write("Paste a YouTube URL, pick a language, and ask questions or get a summary! Using Runnables with strict transcript-only prompts.")
    # Load LLM once
    llm = load_llm()
    # Sidebar for options
    with st.sidebar:
        st.header("Options")
        feature = st.radio("Choose:", ["Summary", "Q&A (Sequential)", "Insights (Parallel)"])
    # Main input area
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input("YouTube URL", placeholder="e.g., https://www.youtube.com/watch?v=...")
    with col2:
        lang_key = st.selectbox("Language", list(SUPPORTED_LANGUAGES.keys()))
        lang_name = SUPPORTED_LANGUAGES[lang_key]
    # Process button
    if url and st.button("Fetch Transcript", type="primary"):
        try:
            with st.spinner("Getting transcript..."):
                transcript = process_video(url, lang_key)
                st.session_state.transcript = transcript # Save in session
                st.session_state.lang = lang_name
                st.success(f"Transcript fetched ({len(transcript)} characters)!")
        except Exception as e:
            st.error(f"Oops! {e}")
            st.info("Try a video with captions enabled.")
    # Show preview if transcript exists
    if "transcript" in st.session_state:
        st.subheader(f"Transcript Preview ({st.session_state.lang})")
        full_text = st.session_state.transcript
        preview = full_text[:500] + "..." if len(full_text) > 500 else full_text
        st.text_area("Preview", preview, height=150, disabled=True)
        st.divider()
        if feature == "Summary":
            st.subheader("Generate Summary (Runnable Chain)")
            if st.button("Create Summary"):
                with st.spinner("Summarizing..."):
                    summary = generate_summary(full_text, llm)
                    st.markdown(summary)
        elif feature == "Q&A (Sequential)":
            st.subheader("Ask a Question (Sequential Chaining)")
            question = st.text_input("Your question:")
            if question and st.button("Get Answer"):
                with st.spinner("Processing..."):
                    answer, condensed = ask_question(full_text, question, llm)
                    st.markdown(f"**Condensed Transcript:** {condensed}")
                    st.divider()
                    st.markdown(f"**Answer:** {answer}")
        else: # Insights (Parallel)
            st.subheader("Generate Insights (Parallel Chaining)")
            if st.button("Create Insights"):
                with st.spinner("Generating in parallel..."):
                    insights = generate_insights(full_text, llm)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Summary:**")
                        st.markdown(insights["summary"])
                    with col2:
                        st.markdown("**Key Entities:**")
                        st.markdown(insights["entities"])

if __name__ == "__main__":
    main()