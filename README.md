# Youtube-rag-application
“Interactive app to upload YouTube videos, generate transcripts with Whisper, and search clips via a Streamlit RAG UI.”
# 🎥 YouTube RAG Application

An interactive Streamlit application that lets you:

- 🔎 Search YouTube or paste a YouTube URL  
- 📝 Auto-transcribe videos using Whisper / WhisperX  
- ✂️ Chunk transcripts with timestamps  
- 🗄️ Upload embeddings to AstraDB through LangFlow  
- 🔍 Perform semantic search over video clips  
- ▶️ Get timestamp-accurate video snippets directly in the UI  

---

# Features

# 1. Video Processing
- Downloads audio using `yt-dlp`
- Transcribes using Whisper (`openai-whisper`)
- Falls back to YouTube captions if Whisper fails
- Optional WhisperX for better word-level timestamps
- Chunks transcript into 800–900 character segments

# 2. RAG Pipeline
- Uses LangFlow's HTTP API
- Uploads chunks to AstraDB vector store  
- Queries embeddings via another LangFlow flow  
- Returns:
  - `video_id`
  - `start` / `end` timestamps
  - `text` snippet
  - `semantic similarity score`

# 3. Streamlit App
Two tabs:

# Upload & Embed**
- Search YouTube (top N videos)  
- OR paste a YouTube URL  
- Transcribes → chunks → sends to LangFlow  

#  Search Videos
- Enter a natural language query  
- Returns the most relevant clips  
- Inline YouTube player with timestamps  



