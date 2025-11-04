# Simple YouTube Q&A App

![YouTube Q&A App](https://via.placeholder.com/800x400?text=YouTube+Q%26A+App) <!-- Placeholder; replace with actual logo if available -->

A lightweight Streamlit application that fetches transcripts from YouTube videos, processes them using LangChain's Runnable chains, and enables users to ask questions, generate summaries, or extract insights. Built with Hugging Face models for on-device inference (CPU-friendly) and strict transcript-only prompting to avoid hallucinations.

## 🚀 Features

- **Transcript Fetching**: Supports English and Hindi transcripts from YouTube videos (manual or auto-generated captions).
- **Q&A (Sequential Chaining)**: Condenses the transcript first, then answers questions using only the provided context. If info isn't in the transcript, it responds "I don't know."
- **Summary Generation**: Creates bullet-point summaries using Runnable chains.
- **Insights (Parallel Chaining)**: Simultaneously generates summaries and extracts key entities (people, places, etc.) from the transcript.
- **Strict Prompting**: All prompts enforce using *only* the transcript—no external knowledge or model biases.
- **Lightweight & Local**: Uses `google/flan-t5-base` model (runs on CPU, no GPU needed). No API keys required.
- **UI**: Clean Streamlit interface with sidebar options, preview, and responsive layout.

## 📋 Prerequisites

- Python 3.9+
- Streamlit for the web app
- YouTube videos with captions enabled (for transcript fetching)

## 🛠️ Installation

1. Clone or download the project:
   ```bash
   git clone <your-repo-url>
   cd simple-youtube-qa-app
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install streamlit youtube-transcript-api langchain-huggingface transformers torch
   ```
   - `torch` is included for Hugging Face pipelines (CPU version by default).
   - Optional: For faster inference, install with CUDA if you have a GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu118`.

4. Run the app:
   ```bash
   streamlit run app.py
   ```
   - Open `http://localhost:8501` in your browser.

## 🔍 How It Works

### Core Components

1. **Transcript Extraction** (`get_transcript`):
   - Uses `youtube_transcript_api` to fetch captions.
   - Prioritizes manual captions in the selected language (EN/HI), falls back to auto-generated or English.
   - Handles errors like disabled transcripts.

2. **Model Loading** (`load_llm`):
   - Loads `google/flan-t5-base` via `HuggingFacePipeline` from LangChain.
   - Cached with `@st.cache_resource` for efficiency.

3. **Prompt Templates**:
   - All use `PromptTemplate` from LangChain.
   - Strict instructions: "Use ONLY the provided transcript. Do not use external knowledge. If not supported, say 'I don't know.'"

4. **Runnable Chains** (No `LLMChain`):
   - **Sequential Chaining** (Q&A): Condense → Answer (using `|` operator).
     - Condense: Reduces transcript to 3-5 key sentences.
     - Answer: Queries the condensed version.
   - **Parallel Chaining** (Insights): Runs summary and entity extraction simultaneously via `RunnableParallel`.
   - **Simple Chain** (Summary): Prompt → LLM → Parser.
   - Output parsing with `StrOutputParser` for clean text.

5. **UI Flow** (Streamlit):
   - Input: YouTube URL + Language.
   - Fetch & Preview transcript.
   - Sidebar: Choose mode (Summary, Q&A Sequential, Insights Parallel).
   - Outputs: Markdown-formatted responses with spinners for UX.

### Example Usage

1. Paste a URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ` (Rickroll for fun).
2. Select language: English.
3. Click "Fetch Transcript" → Preview shows first 500 chars.
4. Choose "Q&A (Sequential)":
   - Question: "What is the main topic?"
   - Output: Condensed transcript + Answer (e.g., "I don't know" if off-topic).
5. Choose "Insights (Parallel)":
   - Two-column output: Summary bullets + Comma-separated entities.

## 📝 Code Structure

- **`app.py`** (Main file):
  - Imports & Config: Languages, patterns for URL parsing.
  - Helpers: `get_video_id`, `get_transcript`, `process_video`.
  - Chains:
    - `ask_question`: Sequential (condense | answer).
    - `generate_insights`: Parallel (summary || entities).
    - `generate_summary`: Simple Runnable.
  - `main()`: Streamlit UI setup, session state for transcript.

- **Key LangChain Concepts Used**:
  - `PromptTemplate`: For all prompts.
  - `RunnablePassthrough` / `RunnableLambda`: Not explicitly, but chains use `|` for sequencing.
  - `RunnableParallel`: For parallel insights.
  - `StrOutputParser`: Cleans LLM outputs.
  - No `LLMChain`—pure Runnables for modularity.

- **Error Handling**:
  - Invalid URLs: ValueError.
  - No transcripts: RuntimeError with user-friendly messages.
  - Spinners & alerts via Streamlit.

## ⚠️ Limitations

- **Model Size**: FLAN-T5-Base is small/fast but may lack nuance for long/complex transcripts. Upgrade to larger models (e.g., FLAN-T5-Large) if needed.
- **Transcript Length**: Full transcripts fed to model; for very long videos (>10k chars), condensation helps but may truncate context.
- **Languages**: Only EN/HI supported (expand `SUPPORTED_LANGUAGES` for more).
- **No GPU**: Defaults to CPU; add `device=0` in pipeline for CUDA.
- **YouTube API**: Relies on public transcripts—private/restricted videos won't work.
- **Hallucinations**: Mitigated by strict prompts, but always verify outputs.

## 🔮 Future Enhancements

- Add more languages/models.
- Integrate vector stores (e.g., FAISS) for RAG on longer transcripts.
- Export summaries as PDF/JSON.
- Authentication for private videos (via YouTube API key).
- Multi-video batch processing.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit changes (`git commit -m 'Add amazing feature'`).
4. Push (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

Questions? Open an issue!

---

*Built with ❤️ using Streamlit, LangChain, and Hugging Face. Last updated: November 2025.*