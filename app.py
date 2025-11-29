import streamlit as st
import requests
import json
import re
import time
import uuid
from pathlib import Path
from transcribe_videos import process_youtube_search, process_single_video

# Configuration
HOST = "https://brainless-vicenta-glottic.ngrok-free.dev"
UPLOAD_FLOW_ID = "6aeffd74-6ad2-48e3-8d94-2ef7af03dfc1"
QUERY_FLOW_ID = "9298cf0a-a1b7-42e3-a04b-f88d4b91610e"
API_KEY = "sk-vZ6VfCG1RMrlURlLKOFlARwLjflZOl5KbJmV3oA8aXE"
CHUNK_SIZE = 900
TIMEOUT = 90.0
OUTPUT_DIR = Path("./whisper_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Page config
st.set_page_config(
    page_title="YouTube RAG Search Engine",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube Video RAG System")
st.markdown("Search through YouTube video transcripts with semantic search")

# ==================== HELPER FUNCTIONS ====================

def attempt_post(url, payload, headers=None, params=None):
    try:
        r = requests.post(url, json=payload, headers=headers or {}, params=params or {}, timeout=TIMEOUT)
        return (200 <= r.status_code < 300), r.status_code, r.text
    except requests.RequestException as e:
        return False, None, f"RequestException: {e}"

def try_auth_methods(url, payload, api_key):
    ok, status, body = attempt_post(url, payload, headers={"Content-Type": "application/json"})
    if ok:
        return ok, status, body, "no-auth"
    if not api_key:
        return ok, status, body, "no-auth-failed"

    ok, status, body = attempt_post(url, payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    })
    if ok:
        return ok, status, body, "bearer"

    ok2, status2, body2 = attempt_post(url, payload, headers={
        "Content-Type": "application/json",
        "x-api-key": api_key
    })
    if ok2:
        return ok2, status2, body2, "x-api-key"

    ok3, status3, body3 = attempt_post(url, payload, headers={
        "Content-Type": "application/json"
    }, params={"api_key": api_key})
    if ok3:
        return ok3, status3, body3, "query-param"

    return ok3, status3, body3, "all-tried"

def clean_text(text):
    cleaned = re.sub(r'\[\d+\.\d+–\d+\.\d+\]', '', text)
    cleaned = ' '.join(cleaned.split())
    return cleaned

def format_for_langflow(doc):
    return f"[VID:{doc['video_id']}|{doc['start']}-{doc['end']}s] {doc['text']}"

def send_chunks_to_langflow(docs, api_key, progress_bar=None, status_text=None):
    """Send document chunks to LangFlow for embedding"""
    total = len(docs)
    results = []
    session_id = str(uuid.uuid4())
    upload_url = f"{HOST}/api/v1/run/{UPLOAD_FLOW_ID}"

    for i, doc in enumerate(docs, 1):
        formatted_text = format_for_langflow(doc)
        payload = {
            "output_type": "text",
            "input_type": "chat",
            "input_value": formatted_text,
            "session_id": session_id
        }

        ok, status, body, method = try_auth_methods(upload_url, payload, api_key)
        results.append((ok, status, doc['video_id']))

        if progress_bar:
            progress_bar.progress(i / total)
        if status_text:
            status_text.text(f"Uploading chunk {i}/{total}...")

        if status in [403, 500]:
            return False, f"Error at chunk {i}: Status {status}"

        time.sleep(0.15)

    successes = sum(1 for r in results if r[0])
    return True, f"Successfully uploaded {successes}/{total} chunks"

def parse_response_json(response_text):
    """Parse JSON array response from LangFlow"""
    try:
        data = json.loads(response_text)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            if 'outputs' in data:
                outputs = data.get('outputs', [])
                if outputs and len(outputs) > 0:
                    first_output = outputs[0]
                    if 'outputs' in first_output:
                        inner = first_output['outputs'][0] if first_output['outputs'] else {}
                        if 'results' in inner:
                            results = inner['results']
                            if 'message' in results:
                                msg = results['message']
                                if isinstance(msg, dict) and 'text' in msg:
                                    return json.loads(msg['text'])
                                elif isinstance(msg, str):
                                    return json.loads(msg)
        return None
    except Exception:
        return None

# ==================== LAYOUT ====================

tab1, tab2 = st.tabs(["📤 Upload & Embed Videos", "🔍 Search Videos"])

# ==================== TAB 1: Upload & Embed ====================
with tab1:
    st.header("Upload Video Transcripts to Database")
    st.markdown("Enter YouTube search query or paste a direct YouTube link to embed into AstraDB")

    input_type = st.radio("Input Method:", ["YouTube Search Query", "Direct YouTube Link"])

    # ----- Option 1: YouTube Search Query -----
    if input_type == "YouTube Search Query":
        query = st.text_input("🔎 YouTube Search Query:", placeholder="e.g., conservation of momentum")
        num_videos = st.number_input("Number of videos to process:", min_value=1, max_value=20, value=5)

        if st.button("🚀 Search, Transcribe & Upload", type="primary"):
            if query:
                with st.spinner(f"Processing {num_videos} videos... This may take several minutes"):
                    st.info(f"🔍 Searching for '{query}' and processing {num_videos} videos...")

                    try:
                        chunks = process_youtube_search(query, int(num_videos))
                        if chunks:
                            progress_bar = st.progress(0.0)
                            status_text = st.empty()
                            success, message = send_chunks_to_langflow(chunks, API_KEY, progress_bar, status_text)
                            if success:
                                st.success("✅ " + message)
                            else:
                                st.error("❌ " + message)
                        else:
                            st.error("❌ No transcript chunks were produced.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.error("❌ Please enter a search query")

    # ----- Option 2: Direct YouTube Link -----
    else:
        st.markdown("### 📺 Paste YouTube Link")
        youtube_link = st.text_input(
            "YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=DxKelGugDa8",
            help="Paste the full YouTube URL or just the video ID"
        )

        video_id = ""
        if youtube_link:
            patterns = [
                r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)',
                r'^([a-zA-Z0-9_-]{11})$'
            ]
            for pattern in patterns:
                match = re.search(pattern, youtube_link)
                if match:
                    video_id = match.group(1)
                    break

            if video_id:
                st.success(f"✅ Detected Video ID: `{video_id}`")
            else:
                st.error("❌ Invalid YouTube URL. Please check the link.")

        if st.button("📥 Transcribe & Upload Video", type="primary", disabled=not video_id):
            if video_id:
                with st.spinner(f"Processing video {video_id}..."):
                    st.info("🎥 Downloading and transcribing video...")

                    try:
                        chunks = process_single_video(video_id)
                        if chunks:
                            progress_bar = st.progress(0.0)
                            status_text = st.empty()
                            success, message = send_chunks_to_langflow(chunks, API_KEY, progress_bar, status_text)
                            if success:
                                st.success(f"✅ Video {video_id} transcribed and uploaded successfully!")
                                st.balloons()
                            else:
                                st.error("❌ " + message)
                        else:
                            st.error("❌ No transcript chunks were produced for this video.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.error("❌ Please enter a valid YouTube link")

# ==================== TAB 2: Search & Retrieve ====================
with tab2:
    st.header("Search Video Database")
    st.markdown("Enter your query to find relevant video clips with timestamps")

    search_query = st.text_input(
        "🔍 Enter your search query:",
        placeholder="e.g., Tell me where conservation of momentum is explained",
        key="search_input"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("🔎 Search", type="primary", use_container_width=True)
    with col2:
        num_results = st.slider("Number of results:", 1, 10, 5)

    if search_button and search_query:
        with st.spinner("Searching database..."):
            try:
                url = f"{HOST}/api/v1/run/{QUERY_FLOW_ID}"
                payload = {
                    "input_value": search_query,
                    "output_type": "text",
                    "input_type": "chat"
                }

                ok, status, body, method = try_auth_methods(url, payload, API_KEY)

                if ok:
                    results = parse_response_json(body)

                    if results:
                        st.success(f"✅ Found {len(results)} relevant video clip(s)")

                        for i, result in enumerate(results[:num_results], 1):
                            video_id = result.get("video_id", "")
                            start = int(result.get("start", 0))
                            end = int(result.get("end", 0))
                            text = result.get("text", "")
                            score = result.get("score", 0)

                            youtube_url = f"https://www.youtube.com/watch?v={video_id}&t={start}s"

                            with st.container():
                                st.markdown(f"### 📹 Clip {i} - Relevance: {score:.2f}")

                                col_a, col_b = st.columns([3, 2])  # wider video, narrower transcript

                                with col_a:
                                    # Embed video in the UI
                                    st.video(youtube_url, start_time=start)
                                    # Optional external link
                                    st.link_button("🔗 Open on YouTube", youtube_url, use_container_width=True)
                                    st.markdown(f"**Video ID:** `{video_id}`")
                                    st.markdown(f"**Time:** {start}s - {end}s ({end-start}s)")

                                with col_b:
                                    clean_text_display = re.sub(r'\[VID:.*?\]\s*', '', text)
                                    snippet = clean_text_display[:300] + "..." if len(clean_text_display) > 300 else clean_text_display
                                    st.markdown("**Transcript Excerpt:**")
                                    st.text_area(
                                        "",
                                        snippet,
                                        height=250,
                                        key=f"text_{i}",
                                        label_visibility="collapsed"
                                    )

                                st.divider()
                    else:
                        st.warning("No results found for your query")
                else:
                    st.error(f"Search failed with status code: {status}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Search timed out. Please try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    elif search_button:
        st.warning("Please enter a search query")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ System Info")
    st.markdown(f"""
    **LangFlow Host:** `{HOST}`

    **Upload Flow ID:**  
    `{UPLOAD_FLOW_ID[:8]}...`

    **Query Flow ID:**  
    `{QUERY_FLOW_ID[:8]}...`

    **Status:** 🟢 Connected
    """)

    st.divider()
    st.markdown("### 📊 Features")
    st.markdown("""
    - Semantic video search
    - Timestamp-accurate results
    - YouTube integration
    - AstraDB vector storage
    - Whisper transcription
    """)

    st.divider()
    st.markdown("### 📁 Output Directory")
    st.code(str(OUTPUT_DIR))
