# transcribe_videos.py

import os, json, tempfile, time
from pathlib import Path

import yt_dlp
import whisper
import torch

try:
    import whisperx
    HAS_WHISPERX = True
except Exception:
    HAS_WHISPERX = False

from youtube_transcript_api import YouTubeTranscriptApi

# CONFIG
NUM_VIDEOS = 5
COOKIEPATH = None
BROWSER_COOKIES = None
WHISPER_MODEL = "base"
MAX_CHARS_PER_CHUNK = 800
OUTPUT_DIR = Path("./whisper_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- SEARCH ----------
def yt_search(query, n=5):
    ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    entries = info.get("entries", []) if info else []
    return [{"id": e.get("id"), "title": e.get("title")} for e in entries if e]

# ---------- DOWNLOAD ----------
def download_audio(video_id, cookiepath=None, browser_cookies=None):
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_dir = tempfile.gettempdir()
    outtmpl = os.path.join(out_dir, f"audio_{video_id}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
    }
    if cookiepath:
        ydl_opts["cookiefile"] = cookiepath
    if browser_cookies:
        ydl_opts["cookies_from_browser"] = browser_cookies

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        for ext in ("mp3", "wav", "m4a", "opus"):
            p = os.path.join(out_dir, f"audio_{video_id}.{ext}")
            if os.path.exists(p):
                return p
        return None
    except Exception:
        return None

# ---------- CAPTIONS ----------
def fetch_captions(video_id):
    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id)
        captions = [{
            "text": e["text"].replace("\n", " ").strip(),
            "start": e["start"],
            "end": e["start"] + e.get("duration", 0)
        } for e in entries]
        full_text = " ".join(c["text"] for c in captions)
        return full_text, captions
    except Exception:
        return None, []

# ---------- WHISPER ----------
def transcribe_audio_whisper(audio_path, model_name=WHISPER_MODEL):
    model = whisper.load_model(model_name, device=device)
    res = model.transcribe(audio_path)
    segments = res.get("segments", [])
    lang = res.get("language", "en")
    return model, segments, lang

def try_whisper_word_timestamps(model, audio_path):
    try:
        out = model.transcribe(audio_path, word_timestamps=True)
        segs = out.get("segments", [])
        if any("words" in s and s["words"] for s in segs):
            return segs
    except Exception:
        pass
    return None

def align_with_whisperx(segments, audio_path, language):
    try:
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        aligned = whisperx.align(segments, align_model, audio_path, device=device, return_seconds=True)
        return aligned.get("segments", [])
    except Exception:
        return None

# ---------- CHUNKING ----------
def segments_to_chunks(aligned_segments, max_chars=MAX_CHARS_PER_CHUNK):
    chunks = []
    cur_text, cur_start, cur_end, cur_word_timings = [], None, None, []

    def flush():
        nonlocal cur_text, cur_start, cur_end, cur_word_timings
        if cur_text:
            chunks.append({
                "text": " ".join(cur_text).strip(),
                "start": cur_start,
                "end": cur_end,
                "word_timings": cur_word_timings
            })
        cur_text, cur_start, cur_end, cur_word_timings = [], None, None, []

    for seg in aligned_segments:
        words = seg.get("words")
        if words:
            for w in words:
                w_text = (w.get("word") or w.get("text") or "").strip()
                if not w_text:
                    continue
                s = float(w.get("start", 0.0))
                e = float(w.get("end", s))
                if cur_start is None:
                    cur_start = s
                cur_end = e
                cur_text.append(w_text)
                cur_word_timings.append({"word": w_text, "start": s, "end": e})
                if len(" ".join(cur_text)) >= max_chars:
                    flush()
        else:
            text = seg.get("text", "")
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", s))
            words_list = text.split()
            for i, w in enumerate(words_list):
                approx_s = s + (i / max(1, len(words_list))) * (e - s)
                approx_e = s + ((i + 1) / max(1, len(words_list))) * (e - s)
                if cur_start is None:
                    cur_start = approx_s
                cur_end = approx_e
                cur_text.append(w)
                cur_word_timings.append({"word": w, "start": approx_s, "end": approx_e})
                if len(" ".join(cur_text)) >= max_chars:
                    flush()
    flush()
    return chunks

# ---------- CORE: one video ----------
def process_video_object(video, cookiepath=None, browser_cookies=None):
    vid = video["id"]
    title = video.get("title", vid)

    audio_path = download_audio(vid, cookiepath=cookiepath, browser_cookies=browser_cookies)
    aligned_segments = []

    if audio_path:
        try:
            model, segments, lang = transcribe_audio_whisper(audio_path)
            wt = try_whisper_word_timestamps(model, audio_path)
            if wt:
                aligned_segments = wt
            else:
                if HAS_WHISPERX:
                    aligned = align_with_whisperx(segments, audio_path, lang)
                    if aligned:
                        aligned_segments = aligned
                    else:
                        aligned_segments = segments
                else:
                    aligned_segments = segments
        except Exception:
            aligned_segments = []

    if not aligned_segments:
        text, caps = fetch_captions(vid)
        if text:
            aligned_segments = [{"text": c["text"], "start": c["start"], "end": c["end"]} for c in caps]
        else:
            return None

    chunks = segments_to_chunks(aligned_segments)

    # Save JSON (optional, keeps your old behavior)
    out_file = OUTPUT_DIR / f"chunks_{vid}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"video_id": vid, "title": title, "chunks": chunks}, f, ensure_ascii=False, indent=2)

    try:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception:
        pass

    # Return chunks in a flat format ready for LangFlow send
    docs = []
    for ch in chunks:
        docs.append({
            "video_id": vid,
            "title": title,
            "start": float(ch.get("start", 0.0)),
            "end": float(ch.get("end", 0.0)),
            "text": ch.get("text", "")
        })
    return docs

# ---------- PUBLIC API ----------
def process_single_video(video_id: str):
    video = {"id": video_id, "title": video_id}
    docs = process_video_object(video, cookiepath=COOKIEPATH, browser_cookies=BROWSER_COOKIES)
    return docs or []

def process_youtube_search(query: str, num_videos: int):
    videos = yt_search(query, n=num_videos)
    all_docs = []
    for v in videos:
        docs = process_video_object(v, cookiepath=COOKIEPATH, browser_cookies=BROWSER_COOKIES)
        if docs:
            all_docs.extend(docs)
    return all_docs
