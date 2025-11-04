import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline
import re

#  Simple config 
SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi"}
# Map to YouTube lang codes
LANG_MAP = {"en": "en", "hi": "hi"}

#  Load the small LLM (cached so it loads once) 
@st.cache_resource
def load_llm():
    """
    Load a small text generation model. Runs on CPU, no GPU needed.
    This is lightweight and fast for short transcripts.
    """
    return pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

#  Helper to extract video ID from URL 
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

#  Get transcript from YouTube (updated for new API) 
def get_transcript(video_id, lang="en"):
    """
    Fetch the video transcript using YouTube's API.
    Tries manual captions first, then auto-generated.
    Returns the full text joined together.
    """
    ytt_api = YouTubeTranscriptApi()  # Instantiate as per new API
    try:
        # List available transcripts (new method: .list(video_id))
        transcript_list = ytt_api.list(video_id)
        
        # Try preferred language
        target_lang = LANG_MAP.get(lang, "en")
        for auto_gen in [False, True]:  # Manual first, then auto
            try:
                if auto_gen:
                    transcript = transcript_list.find_generated_transcript([target_lang])
                else:
                    transcript = transcript_list.find_manually_created_transcript([target_lang])
                
                # Fetch and join text chunks (use .text attribute)
                data = transcript.fetch()
                return " ".join(chunk.text for chunk in data)
            except NoTranscriptFound:
                continue  # Try next option
        
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

#  Process video (main function) 
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

#  Ask the LLM a question 
def ask_question(transcript, question, llm):
    """
    Use the full transcript as context in the prompt.
    Ask the small LLM for an answer.
    """
    prompt = f"""Use only the video transcript below to answer the question.

Transcript: {transcript}

Question: {question}

Answer:"""

    # Generate response (keep it short)
    response = llm(prompt, max_new_tokens=150, do_sample=False)
    answer = response[0]['generated_text'].strip()
    
    # Clean up: Remove the prompt if echoed
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    
    return answer

#  Generate a simple summary 
def generate_summary(transcript, llm):
    """
    Ask the LLM for a bullet-point summary.
    """
    prompt = f"""Summarize the video transcript in 3-5 short bullet points.
Keep each point under 50 words.

Transcript: {transcript}

Summary:
 """

    response = llm(prompt, max_new_tokens=200, do_sample=False)
    summary = response[0]['generated_text'].strip()
    
    # Clean up if needed
    if summary.startswith(prompt):
        summary = summary[len(prompt):].strip()
    
    return summary

# --- Simple Streamlit UI ---
def main():
    # Set up the page
    st.set_page_config(page_title="Simple YouTube Q&A", page_icon="🎥", layout="wide")
    st.title("🎥 Simple YouTube Q&A App")
    st.write("Paste a YouTube URL, pick a language, and ask questions or get a summary!")

    # Load LLM once
    llm = load_llm()

    # Sidebar for options
    with st.sidebar:
        st.header("Options")
        feature = st.radio("Choose:", ["Summary", "Q&A"])  # Simple radio instead of selectbox

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
                st.session_state.transcript = transcript  # Save in session
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
            st.subheader("Generate Summary")
            if st.button("Create Summary"):
                with st.spinner("Summarizing..."):
                    summary = generate_summary(full_text, llm)
                    st.markdown(summary)

        else:  # Q&A
            st.subheader(" Ask a Question")
            question = st.text_input("Your question:")
            if question and st.button("Get Answer"):
                with st.spinner("Thinking..."):
                    answer = ask_question(full_text, question, llm)
                    st.markdown(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()