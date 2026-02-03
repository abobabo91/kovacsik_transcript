import streamlit as st
import openai
import anthropic
import google.generativeai as genai
import os
import tempfile
import subprocess
import sys
import re
from googleapiclient.discovery import build

# Attempt to force upgrade yt-dlp at runtime if it's too old
try:
    import yt_dlp
    version = getattr(yt_dlp.version, '__version__', '0.0.0')
    if version < '2025.01.31':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "yt-dlp"])
        import importlib
        importlib.reload(yt_dlp)
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
    import yt_dlp

from pydub import AudioSegment
from pydub.utils import make_chunks

def download_youtube_audio(url, log_container=None):
    """Downloads YouTube audio and returns the path to the MP3 file."""
    try:
        if log_container:
            log_container.info(f"Using yt-dlp version: {yt_dlp.version.__version__}")
            
        # Get PO Token from session state or secrets
        po_token = st.session_state.get('yt_po_token') or st.secrets.get("youtube", {}).get("PO_TOKEN", None)
        cookies_content = st.session_state.get('yt_cookies')
        
        cookie_file_path = None
        if cookies_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tf:
                tf.write(cookies_content)
                cookie_file_path = tf.name

        ydl_opts = {
            'cookiefile': cookie_file_path,
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(tempfile.gettempdir(), '%(title)s.%(ext)s'),
            'quiet': False,  # Changed to False for debugging
            'no_warnings': False,
            'nocheckcertificate': True,
            'referer': 'https://www.google.com/',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'extractor_args': {
                'youtube': {
                    'player_client': ['web', 'mweb', 'android', 'ios'],
                    'po_token': [f"web+{po_token}", f"mweb+{po_token}", f"android+{po_token}", f"ios+{po_token}"] if po_token else []
                }
            },
            'http_headers': {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            },
            'noplaylist': True, # Ensure we only download one video
        }
        
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(info)
                    mp3_filename = os.path.splitext(filename)[0] + ".mp3"
                    
                    if not os.path.exists(mp3_filename):
                        title = info.get('title', '')
                        if title:
                            possible_path = os.path.join(tempfile.gettempdir(), f"{title}.mp3")
                            if os.path.exists(possible_path):
                                return possible_path
                    return mp3_filename
            except Exception as e:
                if log_container:
                    log_container.error(f"yt-dlp internal error: {e}")
                    with st.expander("Show detailed logs"):
                        st.code(f.getvalue())
                raise e
            finally:
                if cookie_file_path and os.path.exists(cookie_file_path):
                    try:
                        os.remove(cookie_file_path)
                    except:
                        pass
                
    except Exception as e:
        st.error(f"Error downloading YouTube video: {e}")
        return None

def get_playlist_videos(url):
    """Fetches titles and URLs of all videos in a playlist."""
    try:
        # Normalize URL to ensure playlist extraction
        playlist_id_match = re.search(r"list=([a-zA-Z0-9_-]+)", url)
        if playlist_id_match:
            url = f"https://www.youtube.com/playlist?list={playlist_id_match.group(1)}"

        # Get PO Token from session state or secrets
        po_token = st.session_state.get('yt_po_token') or st.secrets.get("youtube", {}).get("PO_TOKEN", None)
        cookies_content = st.session_state.get('yt_cookies')
        
        cookie_file_path = None
        if cookies_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tf:
                tf.write(cookies_content)
                cookie_file_path = tf.name

        ydl_opts = {
            'cookiefile': cookie_file_path,
            'extract_flat': False, # More thorough
            'quiet': False,
            'no_warnings': False,
            'nocheckcertificate': True,
            'ignoreerrors': True, # Skip deleted/unavailable videos
            'cachedir': False,
            'referer': 'https://www.google.com/',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'extractor_args': {
                'youtube': {
                    'player_client': ['web', 'mweb', 'android', 'ios'],
                    'po_token': [f"web+{po_token}", f"mweb+{po_token}", f"android+{po_token}", f"ios+{po_token}"] if po_token else []
                }
            },
            'http_headers': {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            }
        }
        
        import io
        from contextlib import redirect_stdout, redirect_stderr
        f = io.StringIO()
        
        with redirect_stdout(f), redirect_stderr(f):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    if 'entries' in info:
                        entries = [e for e in info['entries'] if e]
                        videos = [{
                            'title': entry.get('title', 'Unknown Title'),
                            'url': f"https://www.youtube.com/watch?v={entry.get('id')}",
                            'id': entry.get('id')
                        } for entry in entries]
                        return videos, f.getvalue()
                    return [], f.getvalue()
            finally:
                if cookie_file_path and os.path.exists(cookie_file_path):
                    try:
                        os.remove(cookie_file_path)
                    except:
                        pass
    except Exception as e:
        return [], f"Error: {str(e)}\n\nLogs:\n{f.getvalue()}"

def get_playlist_videos_api(url, api_key):
    """Fetches titles and URLs of all videos in a playlist using YouTube Data API."""
    try:
        playlist_id_match = re.search(r"list=([a-zA-Z0-9_-]+)", url)
        if not playlist_id_match:
            # Check if it's just a video URL and not a playlist
            video_id_match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})", url)
            if video_id_match:
                video_id = video_id_match.group(1)
                youtube = build('youtube', 'v3', developerKey=api_key)
                request = youtube.videos().list(
                    part="snippet",
                    id=video_id
                )
                response = request.execute()
                if response.get('items'):
                    item = response['items'][0]
                    return [{
                        'title': item['snippet']['title'],
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'id': video_id
                    }], "Single video fetched via API."
            return [], "Could not find playlist ID or video ID in URL."

        playlist_id = playlist_id_match.group(1)
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        videos = []
        next_page_token = None
        
        while True:
            request = youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response.get('items', []):
                video_id = item['snippet']['resourceId']['videoId']
                videos.append({
                    'title': item['snippet']['title'],
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'id': video_id
                })
                
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
        return videos, f"Successfully fetched {len(videos)} videos via API."
    except Exception as e:
        return [], f"API Error: {str(e)}"

# Function to transcribe audio
def transcribe_audio(audio_file, api_key, provider="openai", model="gpt-4o-transcribe", language="en", limit_minutes=0):
    try:
        # Initial trimming logic (shared for both providers)
        processed_audio_file = audio_file
        temp_trimmed_path = None
        
        if limit_minutes > 0:
            try:
                st.warning(f"Trimming audio to first {limit_minutes} minutes...")
                audio = AudioSegment.from_file(audio_file)
                limit_ms = limit_minutes * 60 * 1000
                audio = audio[:limit_ms]
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    temp_trimmed_path = tmp_file.name
                
                audio.export(temp_trimmed_path, format="mp3", bitrate="192k")
                processed_audio_file = open(temp_trimmed_path, "rb")
            except Exception as e:
                st.error(f"Error trimming audio: {e}")
                # Fall back to original file if trimming fails
                processed_audio_file = audio_file

        if provider.lower() == "openai":
            client = openai.OpenAI(api_key=api_key)
            
            # Check file size (OpenAI limit is 25MB)
            processed_audio_file.seek(0, os.SEEK_END)
            file_size = processed_audio_file.tell()
            processed_audio_file.seek(0)
            
            limit_bytes = 10 * 1024 * 1024  # OpenAI limit is actually 25MB, but we use a safer limit for processing
            
            # Determine if we need to split (if duration > 10 mins or size > limit)
            # 10 mins of high quality audio is safe for 25MB limit
            chunk_length_ms = 10 * 60 * 1000
            
            try:
                # We need pydub if it's still too large
                if file_size > limit_bytes:
                    st.warning(f"File size ({file_size / (1024*1024):.2f} MB) exceeds limit. Splitting into chunks...")
                    
                    audio = AudioSegment.from_file(processed_audio_file)
                    full_transcript = ""
                    
                    chunks = make_chunks(audio, chunk_length_ms)
                    st.info(f"Audio duration: {len(audio)/60000:.1f} mins. Split into {len(chunks)} chunks for transcription.")
                    
                    progress_bar = st.progress(0)
                    
                    for i, chunk in enumerate(chunks):
                        chunk_temp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                                chunk_temp_path = tmp_file.name
                            
                            chunk.export(chunk_temp_path, format="mp3", bitrate="192k")
                            
                            with open(chunk_temp_path, "rb") as audio_chunk_file:
                                response = client.audio.transcriptions.create(
                                    model=model,
                                    file=audio_chunk_file,
                                    language=language
                                )
                                full_transcript += response.text + " "
                                
                        finally:
                            if chunk_temp_path and os.path.exists(chunk_temp_path):
                                os.unlink(chunk_temp_path)
                        
                        progress_bar.progress((i + 1) / len(chunks))
                    
                    return full_transcript.strip()
                else:
                    # Process as single file
                    response = client.audio.transcriptions.create(
                        model=model,
                        file=processed_audio_file,
                        language=language
                    )
                    return response.text
            finally:
                # If we opened a new file for the trimmed version, close it
                if temp_trimmed_path and processed_audio_file != audio_file:
                    processed_audio_file.close()
                    if os.path.exists(temp_trimmed_path):
                        os.unlink(temp_trimmed_path)
            
        elif provider.lower() == "google":
            genai.configure(api_key=api_key)
            
            # Create a temporary file to handle the uploaded audio data
            suffix = f".{audio_file.name.split('.')[-1]}" if hasattr(audio_file, 'name') else ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                if hasattr(processed_audio_file, 'read'):
                    processed_audio_file.seek(0)
                    tmp_file.write(processed_audio_file.read())
                else:
                    tmp_file.write(processed_audio_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Upload the file to Gemini
                myfile = genai.upload_file(tmp_file_path)
                
                # Map language code to full name for better prompting
                lang_map = {"en": "English", "hu": "Hungarian"}
                lang_full = lang_map.get(language, "English")

                # Generate content using the audio file
                model_instance = genai.GenerativeModel(model)
                result = model_instance.generate_content([
                    myfile, 
                    f"Generate a transcript of the speech. The language is {lang_full}."
                ])
                
                return result.text
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                
                # If we opened a new file for the trimmed version, close it
                if temp_trimmed_path and processed_audio_file != audio_file:
                    processed_audio_file.close()
                    if os.path.exists(temp_trimmed_path):
                        os.unlink(temp_trimmed_path)
                    
        else:
            st.error(f"Unsupported provider: {provider}")
            return None
            
    except Exception as e:
        st.error(f"Error transcribing audio with {provider}: {e}")
        return None

# Function to format transcription
def format_transcription(transcription, api_key, provider, model):
    if not transcription:
        return ""
    
    formatting_prompt = f"""I want a transcript of an interview. 
    I give you the continuous text, and your job is to make it a coherent interview text, without changing any wording. 
    Just change the structure and keep the wording. Here is the text:
{transcription}"""
    
    try:
        if provider.lower() == "claude":
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=model,
                max_tokens=8000,
                temperature=0,
                system="",
                messages=[{"role": "user", "content": formatting_prompt}]
            )
            return message.content[0].text

        elif provider.lower() == "openai":
            client = openai.OpenAI(api_key=api_key)
            tokens_param = "max_completion_tokens" if any(m in model for m in ["o1", "o3", "gpt-5"]) else "max_tokens"
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": formatting_prompt}
                ],
                **{tokens_param: 8000}
            )
            return response.choices[0].message.content

        elif provider.lower() == "google":
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(formatting_prompt)
            return response.text

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except Exception as e:
        st.error(f"Error formatting transcription with {provider}: {e}")
        return ""
    
def summarize_transcription(structured_transcription, api_key, summary_prompt, provider="claude", model=None):
   
    if not structured_transcription:
        return ""
    
    prompt = f"""{summary_prompt}
{structured_transcription}"""
    
    try:
        if provider.lower() == "claude":
            # Default to latest Claude model if none specified
            claude_model = model or "claude-sonnet-4-5-20250929"
            
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=claude_model,
                max_tokens=8000,
                temperature=0,
                system="",
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
            
        elif provider.lower() == "openai":
            from openai import OpenAI

            openai_model = model or "gpt-5.1"
            client = OpenAI(api_key=api_key)
            
            # Determine whether to use max_tokens or max_completion_tokens (newer models prefer the latter)
            tokens_param = "max_completion_tokens" if any(m in openai_model for m in ["o1", "o3", "gpt-5"]) else "max_tokens"
            
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ],
                **{tokens_param: 8000} 
            )
            
            return response.choices[0].message.content
            
        elif provider.lower() == "google":
            genai.configure(api_key=api_key)
            google_model_name = model or "gemini-3-pro-preview"
            
            model_instance = genai.GenerativeModel(google_model_name)
            response = model_instance.generate_content(prompt)
            return response.text

        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'claude', 'openai', or 'google'.")
            
    except Exception as e:
        st.error(f"Error summarizing transcription with {provider}: {e}")
        return ""

# Page config for better appearance
# st.set_page_config(page_title="Interview Transcription Tool", layout="wide")

st.title("Interview Transcription and Summarization")

# --- API KEYS ---
# Load API keys from Streamlit secrets
openai_api_key = st.secrets.get("openai", {}).get("OPENAI_API_KEY", None)
claude_api_key = st.secrets.get("anthropic", {}).get("ANTHROPIC_API_KEY", None)
google_api_key = st.secrets.get("google", {}).get("GEMINI_API_KEY", None)
youtube_api_key = st.secrets.get("youtube", {}).get("YOUTUBE_API_KEY", None)

# Display warning if keys are missing (optional, but helpful for debugging)
if not openai_api_key:
    st.sidebar.warning("OpenAI API Key not found in secrets (`[openai] OPENAI_API_KEY`).")
if not claude_api_key:
    st.sidebar.warning("Claude API Key not found in secrets (`[anthropic] ANTHROPIC_API_KEY`).")
if not google_api_key:
    st.sidebar.warning("Google Gemini API Key not found in secrets (`[google] GEMINI_API_KEY`).")
if not youtube_api_key:
    st.sidebar.warning("YouTube API Key not found in secrets (`[youtube] YOUTUBE_API_KEY`).")

# Initialize session state variables
if 'raw_transcription' not in st.session_state:
    st.session_state.raw_transcription = ''
if 'structured_transcription' not in st.session_state:
    st.session_state.structured_transcription = ''
if 'structured_summary' not in st.session_state:
    st.session_state.structured_summary = ''
if 'last_processed_file' not in st.session_state:
    st.session_state.last_processed_file = None
if 'summary_model_provider' not in st.session_state:
    st.session_state.summary_model_provider = "Google"
# summary_selected_model removed in favor of dynamic resolution
if 'summary_prompt' not in st.session_state:
    st.session_state.summary_prompt = """I give you the transcription of an interview. It is a customer discovery call about a company, exploring what they do, their business needs, and their methods.

What I need is a summary of the interview for my notes. 
Go question by question. Write the summary of the question and write the answers into bullet points. Shorten every answer, use keywords when possible.
Use exact wording when it matters. Do not add extra wording, but only what has been said.
I want to keep every relevant piece of information that has been said. Focus on not losing any meaningful information.


For example from this transcription:

"
**Interviewer:** Context... is this something that you actually do regularly after you search and analyze?

**Respondent:** Unfortunately. *[Laughs]*

**Interviewer:** Okay. You know, it's so funny when you're a startup founder and you talk to people and they say something like that—"unfortunately"—you would want to feel bad for them, but at the same time, you always get a little excited because there might be something that you can solve for them. And that's... that's good to hear, really. So, yeah, can you maybe elaborate a bit? What do you mean by "unfortunately"? What's the painstaking part?

**Respondent:** It's just a lot of work that's not very exciting. You first have to search, then you have to go through the articles or the patent applications. It's just a lot of work and it's not very exciting. That's the problem. Also, the quality... it can get expensive very quickly. Because you have to review a lot of documents. So it's always this balance that you need to find. There's a certain amount of budget that people want to spend. There are people that spend 5k on a Freedom to Operate (FTO) analysis and there are people that spend 100,000 Euros on a Freedom to Operate analysis.

Obviously, you have both a novelty search and an FTO analysis, but for one, it's so much more work that you do than the other. The other one [novelty search] is just really looking at some feature of whatever they're doing, and you specifically look for that field. But that doesn't necessarily mean that there's not an application that might be limiting their freedom to operate that has a slightly different classification, or a slightly different terminology, or something that you will not pick up. Also, the search is usually not *that* problematic; it's just really analyzing the documents themselves. That's really the annoying bit. Some people really like it; I am not really that big a fan.

**Interviewer:** Yeah.

**Respondent:** Yeah, because it's so tedious and you always end up with the feeling that you haven't found everything.

**Interviewer:** Oh yeah, yeah, I know that feeling. I’ve heard that before. It never really feels exhaustive or complete. There is always a possibility of missing something.

**Respondent:** Yeah. So it's also not very fulfilling when you're done. It's never really complete.
"

The following would be an ideal output, in this exact format (no bold font):

"
- Is FTO search and analysis something you do regularly?
    - Yes, unfortunately
- Why "unfortunately"? What's the painful part?
    - Lot of work, not very exciting
    - First have to search, then go through patent applications
    - Gets expensive very quickly
    - Need to review many documents
    - Have to navigate budget constraints
    - Lower budget = more limited search = higher risk of missing relevant patents with different classification/terminology
    - Analysis is the most problematic part (not the search itself)
    - That’s very tedious/annoying
    - Never feels complete, always possibility of missing something
"

This is the transcript I want you to process:"""

# YouTube Pipeline Session State
if 'yt_results' not in st.session_state:
    st.session_state.yt_results = {} # video_id -> {title, raw, coherent, detailed, short}
if 'playlist_videos' not in st.session_state:
    st.session_state.playlist_videos = []
if 'last_playlist_url' not in st.session_state:
    st.session_state.last_playlist_url = ''
if 'current_view_video_id' not in st.session_state:
    st.session_state.current_view_video_id = None

if 'formatting_provider' not in st.session_state:
    st.session_state.formatting_provider = "Google"
if 'formatting_model' not in st.session_state:
    st.session_state.formatting_model = "gemini-3-pro-preview"

def get_summary_model(provider):
    """Helper to resolve the summary model based on provider and session state."""
    if provider == "Claude":
        return st.session_state.get("summary_model_claude", "claude-sonnet-4-5-20250929")
    elif provider == "OpenAI":
        return st.session_state.get("summary_model_openai", "gpt-5.1")
    elif provider == "Google":
        return st.session_state.get("summary_model_google", "gemini-3-pro-preview")
    return None

# Create tabs for a better UI experience
tab1, tab2, tab3 = st.tabs(["Transcribe & Format", "Summarize", "YouTube Transcribe"])

with tab1:
    st.header("Get Interview Transcription")
    
    # Create radio buttons for input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload Audio File", "Enter Text Directly"]
    )
    
    if input_method == "Upload Audio File":
        st.subheader("Audio Model Settings")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            language_options = {"English": "en", "Hungarian": "hu"}
            selected_language_label = st.selectbox(
                "Language",
                list(language_options.keys()),
                index=0
            )
            selected_language = language_options[selected_language_label]

        with col2:
            transcription_provider = st.selectbox(
                "Transcription Provider",
                ["Google", "OpenAI"],
                index=0
            )
            
        with col3:
            if transcription_provider == "Google":
                transcription_model = st.selectbox(
                    "Google Model",
                    ["gemini-3-pro-preview", "gemini-2.5-flash"],
                    index=0,
                    help="Gemini 3 Pro is high quality, Gemini 2.5 Flash is fast and efficient."
                )
                t_api_key = google_api_key
            else:
                transcription_model = st.selectbox(
                    "OpenAI Model",
                    ["gpt-4o-transcribe", "whisper-1"],
                    index=0,
                    help="gpt-4o-transcribe is generally more accurate than whisper-1."
                )
                t_api_key = openai_api_key

        with col4:
            limit_minutes = st.number_input(
                "Limit Minutes (0=Full)", 
                min_value=0, 
                value=0,
                step=1,
                help="Set to 0 to transcribe the entire file. Useful for testing or saving costs."
            )

        
        # Formatting Options
        st.subheader("Language Model Settings")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.session_state.formatting_provider = st.selectbox(
                "Formatting Provider",
                ["Google", "Claude", "OpenAI"],
                index=0,
                key="fmt_provider_select_upload"
            )
        
        with f_col2:
            if st.session_state.formatting_provider == "Google":
                st.session_state.formatting_model = st.selectbox(
                    "Formatting Model",
                    ["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash"],
                    index=0,
                    key="fmt_model_select_upload"
                )
                f_api_key = google_api_key
            elif st.session_state.formatting_provider == "Claude":
                st.session_state.formatting_model = st.selectbox(
                    "Formatting Model",
                    ["claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"],
                    index=0,
                    format_func=lambda x: x.split("-2025")[0].replace("-", " ").title(),
                    key="fmt_model_select_upload"
                )
                f_api_key = claude_api_key
            else: # OpenAI
                st.session_state.formatting_model = st.selectbox(
                    "Formatting Model",
                    ["gpt-5.1", "gpt-4.1", "gpt-4o", "o3-mini", "o1"],
                    index=0,
                    key="fmt_model_select_upload"
                )
                f_api_key = openai_api_key

        st.subheader("Upload an audio file")

        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "m4a", "wav"], label_visibility="collapsed")
        
        start_processing = st.button("Start Transcription")

        # Create containers for progressive loading
        raw_container = st.empty()
        formatted_container = st.empty()
        summary_container = st.empty()

        # Helper functions to display content
        def display_raw(container):
             if st.session_state.raw_transcription:
                with container.container():
                    st.subheader("Raw Transcription")
                    if st.button("Rerun Transcription", key="rerun_trans"):
                        if not uploaded_file:
                             st.error("No file uploaded.")
                        else:
                            with st.spinner(f"Transcribing with {transcription_provider}..."):
                                uploaded_file.seek(0)
                                new_raw = transcribe_audio(
                                    uploaded_file, 
                                    t_api_key, 
                                    transcription_provider, 
                                    transcription_model, 
                                    selected_language, 
                                    limit_minutes
                                )
                                if new_raw:
                                    st.session_state.raw_transcription = new_raw
                                    st.success("Transcription updated!")
                                    st.rerun()

                    st.text_area("Raw transcription text", st.session_state.raw_transcription, height=200, key="raw_text_area_disp")
                    
        def display_formatted(container):
            if st.session_state.structured_transcription:
                with container.container():
                    st.subheader("Formatted Interview")
                    if st.button("Rerun Formatting", key="rerun_fmt"):
                        with st.spinner(f"Formatting with {st.session_state.formatting_provider}..."):
                             if not f_api_key:
                                 st.error(f"Missing API Key for {st.session_state.formatting_provider}")
                             else:
                                new_fmt = format_transcription(
                                    st.session_state.raw_transcription, 
                                    f_api_key, 
                                    st.session_state.formatting_provider, 
                                    st.session_state.formatting_model
                                )
                                if new_fmt:
                                    st.session_state.structured_transcription = new_fmt
                                    st.success("Formatting updated!")
                                    st.rerun()

                    st.text_area("Formatted interview text", st.session_state.structured_transcription, height=300, key="fmt_text_area_disp")

        def display_summary(container):
            if st.session_state.structured_summary:
                with container.container():
                    st.subheader("Interview Summary")
                    
                    # Determine keys/providers for summary from session state defaults
                    summ_provider = st.session_state.summary_model_provider
                    summ_model = get_summary_model(summ_provider)
                    
                    # Get appropriate API key for summary
                    summ_api_key = None
                    if summ_provider == "Claude":
                        summ_api_key = claude_api_key
                    elif summ_provider == "OpenAI":
                        summ_api_key = openai_api_key
                    elif summ_provider == "Google":
                        summ_api_key = google_api_key
                    
                    if st.button("Rerun Summary", key="rerun_sum"):
                         with st.spinner(f"Summarizing with {summ_provider} ({summ_model})..."):
                            if not summ_api_key:
                                st.error(f"Missing API Key for {summ_provider}")
                            else:
                                new_sum = summarize_transcription(
                                    st.session_state.structured_transcription, 
                                    summ_api_key,
                                    st.session_state.summary_prompt,
                                    provider=summ_provider,
                                    model=summ_model
                                )
                                if new_sum:
                                    st.session_state.structured_summary = new_sum
                                    st.success("Summary updated!")
                                    st.rerun()

                    st.text_area("Summary", st.session_state.structured_summary, height=400, key="sum_text_area_disp")

        processing_happened = False

        if start_processing:
            if not uploaded_file:
                st.warning("There is no file uploaded.")
            else:
                processing_happened = True
                if not t_api_key:
                    st.error(f"Missing {transcription_provider} API Key. Please add it to your .streamlit/secrets.toml file.")
                else:
                    status_container = st.empty()
                    try:
                        # 1. Transcribe
                        status_container.info(f"Step 1/3: Transcribing audio with {transcription_provider} ({transcription_model})...")
                        # Reset file pointer just in case
                        uploaded_file.seek(0)
                        
                        raw_transcription = transcribe_audio(uploaded_file, t_api_key, transcription_provider, transcription_model, selected_language, limit_minutes)
                        
                        if raw_transcription:
                            st.session_state.raw_transcription = raw_transcription
                            display_raw(raw_container) # Display immediately
                            
                            # 2. Format
                            status_container.info(f"Step 2/3: Formatting transcription with {st.session_state.formatting_provider}...")
                            if not f_api_key:
                                st.warning(f"Missing {st.session_state.formatting_provider} API Key. Skipping formatting and summarization.")
                            else:
                                formatted_transcription = format_transcription(
                                    raw_transcription, 
                                    f_api_key, 
                                    st.session_state.formatting_provider, 
                                    st.session_state.formatting_model
                                )
                                if formatted_transcription:
                                    st.session_state.structured_transcription = formatted_transcription
                                    display_formatted(formatted_container) # Display immediately
                                    
                                    # 3. Summarize
                                    status_container.info("Step 3/3: Generating summary...")
                                    
                                    # Determine keys/providers for summary from session state defaults
                                    summ_provider = st.session_state.summary_model_provider
                                    summ_model = get_summary_model(summ_provider)
                                    
                                    # Get appropriate API key for summary
                                    summ_api_key = None
                                    if summ_provider == "Claude":
                                        summ_api_key = claude_api_key
                                    elif summ_provider == "OpenAI":
                                        summ_api_key = openai_api_key
                                    elif summ_provider == "Google":
                                        summ_api_key = google_api_key
                                        
                                    if not summ_api_key:
                                        st.warning(f"Missing {summ_provider} API Key. Skipping summarization.")
                                    else:
                                        summary = summarize_transcription(
                                            formatted_transcription, 
                                            summ_api_key,
                                            st.session_state.summary_prompt,
                                            provider=summ_provider,
                                            model=summ_model
                                        )
                                        if summary:
                                            st.session_state.structured_summary = summary
                                            st.session_state.last_processed_file = uploaded_file.name
                                            display_summary(summary_container) # Display immediately
                                            status_container.success("Processing Complete! (Transcription -> Formatting -> Summary)")
                                        else:
                                            status_container.error("Summarization failed.")
                                else:
                                    status_container.error("Formatting failed.")
                        else:
                            status_container.error("Transcription failed.")
                            
                    except Exception as e:
                        status_container.error(f"An error occurred during processing: {e}")

        # If we didn't process just now, display existing content
        if not processing_happened:
            display_raw(raw_container)
            display_formatted(formatted_container)
            display_summary(summary_container)

    else:  # "Enter Text Directly"
        # Formatting Options for Text Input
        st.subheader("Language Model Settings")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.session_state.formatting_provider = st.selectbox(
                "Formatting Provider",
                ["Google", "Claude", "OpenAI"],
                index=0,
                key="fmt_provider_select_text"
            )
        
        with f_col2:
            if st.session_state.formatting_provider == "Google":
                st.session_state.formatting_model = st.selectbox(
                    "Formatting Model",
                    ["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash"],
                    index=0,
                    key="fmt_model_select_text"
                )
                f_api_key = google_api_key
            elif st.session_state.formatting_provider == "Claude":
                st.session_state.formatting_model = st.selectbox(
                    "Formatting Model",
                    ["claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"],
                    index=0,
                    format_func=lambda x: x.split("-2025")[0].replace("-", " ").title(),
                    key="fmt_model_select_text"
                )
                f_api_key = claude_api_key
            else: # OpenAI
                st.session_state.formatting_model = st.selectbox(
                    "Formatting Model",
                    ["gpt-5.1", "gpt-4.1", "gpt-4o", "o3-mini", "o1"],
                    index=0,
                    key="fmt_model_select_text"
                )
                f_api_key = openai_api_key

        raw_transcription_input = st.text_area(
            "Enter your raw transcription text here:",
            height=250,
            value=st.session_state.raw_transcription
        )
        
        if st.button("Use This Text"):
            st.session_state.raw_transcription = raw_transcription_input
            st.success("Text saved as raw transcription!")
    


with tab2:
    st.header("Summarize Interview")
    
    if not st.session_state.structured_transcription:
        st.info("Please complete the transcription and formatting steps first.")
    else:
        st.subheader("Customize Summary Prompt")
        
        # Let user edit the summary prompt
        st.session_state.summary_prompt = st.text_area(
            "Edit the summary prompt below. Hit Ctrl + Enter to save.",
            value=st.session_state.summary_prompt,
            height=300
        )
        
        # Model selection options
        st.subheader("Select AI Model for Summarization")
        
        col_prov, col_mod = st.columns(2)
        
        with col_prov:
            # Use session state to persist selection
            model_provider = st.radio(
                "AI Provider", 
                ["Google", "Claude", "OpenAI"],
                key="summary_model_provider"
            )
        
        with col_mod:
            # Display appropriate model options based on provider
            if model_provider == "Google":
                # Latest Anthropic Models (Late 2025)
                # Latest Google Models (Late 2025)
                st.selectbox(
                    "Select Google Model",
                    ["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash"],
                    index=0,
                    key="summary_model_google"
                )
                api_key = google_api_key
                provider = "google"
                
            elif model_provider == "OpenAI":
                # Latest OpenAI Models (Late 2025)
                st.selectbox(
                    "Select OpenAI Model",
                    ["gpt-5.1", "gpt-4.1", "gpt-4o", "o3-mini", "o1"],
                    index=0,
                    key="summary_model_openai"
                )
                api_key = openai_api_key
                provider = "openai"
                
            else:  # Claude
                st.selectbox(
                    "Select Claude Model",
                    [
                        "claude-sonnet-4-5-20250929", 
                        "claude-opus-4-5-20251101", 
                        "claude-haiku-4-5-20251001"
                    ],
                    index=0,
                    format_func=lambda x: x.split("-2025")[0].replace("-", " ").title(), # Beautify names
                    key="summary_model_claude"
                )
                api_key = claude_api_key
                provider = "claude"

        # Determine current model for the button action
        current_model = get_summary_model(model_provider)

        # Button to generate/regenerate summary
        if st.button("Regenerate Summary"):
            if not api_key:
                 st.error(f"Missing {model_provider} API Key. Please add it to your .streamlit/secrets.toml file.")
            else:
                with st.spinner(f"Generating summary with {current_model}..."):
                    st.session_state.summary_prompt = st.session_state.summary_prompt  # Save the edited prompt
                    st.session_state.structured_summary = summarize_transcription(
                        st.session_state.structured_transcription, 
                        api_key,
                        st.session_state.summary_prompt,
                        provider=provider,
                        model=current_model
                    )
                    if st.session_state.structured_summary:
                        st.success("Summary generated!")
        
        # Display the summary
        if st.session_state.structured_summary:
            st.subheader("Interview Summary")
            st.text_area("Summary", st.session_state.structured_summary, height=400)

with tab3:
    st.header("YouTube Video Transcription")
    
    col_url, col_fetch_dlp, col_fetch_api = st.columns([3, 1, 1])
    with col_url:
        youtube_url = st.text_input("Enter YouTube Video or Playlist URL:", placeholder="https://www.youtube.com/watch?v=... or https://www.youtube.com/playlist?list=...", label_visibility="collapsed")
    
    # URL Type Detection
    is_playlist = "list=" in youtube_url if youtube_url else False
    if youtube_url:
        if is_playlist:
            st.info("This looks like a playlist. Please use 'Fetch' to list videos and select which ones to process.")
        else:
            st.success("Single video detected. You can download or transcribe it immediately.")

    with st.expander("Advanced YouTube Settings (Cookies / PO Token)"):
        st.info("""
        If you encounter **HTTP Error 403: Forbidden**, YouTube may be blocking the server's IP. 
        Providing cookies or a PO Token can help bypass this.
        
        **How to obtain Cookies:**
        1. Install a browser extension like 'Get cookies.txt LOCALLY' or 'EditThisCookie'.
        2. Log into YouTube in your browser.
        3. Use the extension to export cookies for `youtube.com` in **Netscape** format.
        4. Paste the entire text content below.
        
        **How to obtain a PO Token:**
        1. Use a tool like [yt-dlp-get-pot](https://github.com/coletdjnz/yt-dlp-get-pot-provider) or follow the [PO Token Guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide).
        2. Alternatively, some browser extensions for `yt-dlp` can generate these.
        3. Paste the token below.
        """)
        st.session_state.yt_cookies = st.text_area("YouTube Cookies (Netscape format text):", help="Paste the content of your exported cookies.txt file here.")
        st.session_state.yt_po_token = st.text_input("Manual PO Token:", help="Enter a Proof of Origin token to bypass bot detection.")

    with col_fetch_dlp:
        fetch_clicked = st.button("Fetch (yt-dlp)", use_container_width=True)
    with col_fetch_api:
        fetch_api_clicked = st.button("Fetch (API)", use_container_width=True)

    st.subheader("Transcription Settings")
    yt_col1, yt_col2, yt_col3 = st.columns(3)
    
    with yt_col1:
        yt_language_options = {"English": "en", "Hungarian": "hu"}
        yt_selected_language_label = st.selectbox(
            "Language",
            list(yt_language_options.keys()),
            index=0,
            key="yt_lang"
        )
        yt_selected_language = yt_language_options[yt_selected_language_label]

    with yt_col2:
        yt_transcription_provider = st.selectbox(
            "Transcription Provider",
            ["Google", "OpenAI"],
            index=0,
            key="yt_prov"
        )
        
    with yt_col3:
        if yt_transcription_provider == "Google":
            yt_transcription_model = st.selectbox(
                "Google Model",
                ["gemini-3-pro-preview", "gemini-2.5-flash"],
                index=0,
                key="yt_mod_goog"
            )
            yt_t_api_key = google_api_key
        else:
            yt_transcription_model = st.selectbox(
                "OpenAI Model",
                ["gpt-4o-transcribe", "whisper-1"],
                index=0,
                key="yt_mod_oa"
            )
            yt_t_api_key = openai_api_key

    yt_limit_minutes = st.number_input(
        "Limit Minutes (0=Full)", 
        min_value=0, 
        value=0,
        step=1,
        key="yt_limit"
    )

    # Fetch Playlist/Video Info logic
    if fetch_clicked:
        if not youtube_url:
            st.warning("Please enter a URL first.")
        else:
            with st.spinner("Fetching YouTube metadata via yt-dlp..."):
                videos, logs = get_playlist_videos(youtube_url)
                if videos:
                    st.session_state.playlist_videos = videos
                    st.session_state.last_playlist_url = youtube_url
                    if len(videos) == 1:
                        st.session_state[f"check_{videos[0]['id']}"] = True
                    st.success(f"Found {len(videos)} videos.")
                else:
                    st.error("No videos found. Check URL or logs below.")
                    with st.expander("Show Fetch Logs"):
                        st.code(logs)
                    # Fallback for single video if playlist fetch failed but it might be a single video
                    st.session_state.playlist_videos = [{'title': 'Single Video (Manual)', 'url': youtube_url, 'id': 'single'}]

    if fetch_api_clicked:
        if not youtube_url:
            st.warning("Please enter a URL first.")
        elif not youtube_api_key:
            st.error("YouTube API Key not found in secrets. Please add it to `.streamlit/secrets.toml` under `[youtube] YOUTUBE_API_KEY`.")
        else:
            with st.spinner("Fetching YouTube metadata via API..."):
                videos, logs = get_playlist_videos_api(youtube_url, youtube_api_key)
                if videos:
                    st.session_state.playlist_videos = videos
                    st.session_state.last_playlist_url = youtube_url
                    if len(videos) == 1:
                        st.session_state[f"check_{videos[0]['id']}"] = True
                    st.success(f"Found {len(videos)} videos.")
                else:
                    st.error(f"Failed to fetch videos via API: {logs}")

    selected_videos = []
    
    # Populate selected_videos based on URL type or fetched list
    if youtube_url and not is_playlist:
        # For single videos, bypass selection
        video_id = "single"
        video_id_match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})", youtube_url)
        if video_id_match:
            video_id = video_id_match.group(1)
            
        selected_videos = [{
            'title': 'Single Video',
            'url': youtube_url,
            'id': video_id
        }]
    elif st.session_state.playlist_videos:
        st.subheader("Select videos to process:")
        
        # Select all / Deselect all logic
        col_all1, col_all2 = st.columns(2)
        with col_all1:
            if st.button("Select All"):
                for v in st.session_state.playlist_videos:
                    st.session_state[f"check_{v['id']}"] = True
        with col_all2:
            if st.button("Deselect All"):
                for v in st.session_state.playlist_videos:
                    st.session_state[f"check_{v['id']}"] = False

        # Checkbox list
        for video in st.session_state.playlist_videos:
            is_checked = st.checkbox(video['title'], key=f"check_{video['id']}")
            if is_checked:
                selected_videos.append(video)

    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        if st.button("Download MP3", use_container_width=True):
            if not selected_videos:
                st.warning("No videos selected. Fetch info and select videos first.")
            else:
                for video in selected_videos:
                    status = st.empty()
                    with st.spinner(f"Downloading {video['title']}..."):
                        mp3_path = download_youtube_audio(video['url'], log_container=status)
                        if mp3_path and os.path.exists(mp3_path):
                            with open(mp3_path, "rb") as f:
                                st.download_button(
                                    label=f"Click here to save {video['title']}",
                                    data=f,
                                    file_name=os.path.basename(mp3_path),
                                    mime="audio/mpeg",
                                    key=f"dl_{video['id']}_{os.urandom(4).hex()}"
                                )
                        else:
                            st.error(f"Failed to download {video['title']}.")

    with btn_col2:
        if st.button("Transcribe Selected Videos", use_container_width=True):
            if not selected_videos:
                st.warning("No videos selected. Fetch info and select videos first.")
            elif not yt_t_api_key:
                st.error(f"Missing API key for {yt_transcription_provider}.")
            else:
                for i, video in enumerate(selected_videos):
                    yt_status = st.empty()
                    try:
                        prefix = f"[{i+1}/{len(selected_videos)}] {video['title']}: "
                        yt_status.info(prefix + "Step 1/5: Downloading audio...")
                        mp3_path = download_youtube_audio(video['url'], log_container=yt_status)
                        
                        if mp3_path and os.path.exists(mp3_path):
                            with open(mp3_path, "rb") as audio_file:
                                # 2. Transcribe
                                yt_status.info(prefix + f"Step 2/5: Transcribing with {yt_transcription_provider}...")
                                raw_trans = transcribe_audio(
                                    audio_file, 
                                    yt_t_api_key, 
                                    yt_transcription_provider, 
                                    yt_transcription_model, 
                                    yt_selected_language, 
                                    yt_limit_minutes
                                )
                                
                                if raw_trans:
                                    # 3. Coherence
                                    yt_status.info(prefix + "Step 3/5: Making text coherent...")
                                    f_prov = st.session_state.formatting_provider
                                    f_mod = st.session_state.formatting_model
                                    f_key = google_api_key if f_prov == "Google" else (claude_api_key if f_prov == "Claude" else openai_api_key)

                                    if f_key:
                                        coherence_instr = "I give you a transcript. Your job is to organize it into logical paragraphs based on the topics discussed. Correct any obvious transcription errors in names or technical terms, but keep the original wording exactly as is. Do not add any extra commentary."
                                        coherent_text = summarize_transcription(raw_trans, f_key, coherence_instr, provider=f_prov.lower(), model=f_mod)
                                        
                                        # 4. Detailed Summary
                                        yt_status.info(prefix + "Step 4/5: Generating detailed summary...")
                                        detailed_prompt = "Provide a detailed summary of this transcript. Keep every meaningful aspect that has been mentioned. Use bullet points."
                                        detailed_summary = summarize_transcription(coherent_text, f_key, detailed_prompt, provider=f_prov.lower(), model=f_mod)
                                        
                                        # 5. Short Summary
                                        yt_status.info(prefix + "Step 5/5: Generating short summary...")
                                        short_prompt = "Provide a summary of at most 3 paragraphs. Focus on the most important information and key takeaways."
                                        short_summary = summarize_transcription(coherent_text, f_key, short_prompt, provider=f_prov.lower(), model=f_mod)
                                        
                                        # Store in results dictionary
                                        video_id = video['id']
                                        st.session_state.yt_results[video_id] = {
                                            'title': video['title'],
                                            'raw': raw_trans,
                                            'coherent': coherent_text,
                                            'detailed': detailed_summary,
                                            'short': short_summary
                                        }
                                        st.session_state.current_view_video_id = video_id
                                        yt_status.success(prefix + "Done!")
                                    else:
                                        yt_status.error(prefix + "Missing formatting API key.")
                                else:
                                    yt_status.error(prefix + "Transcription failed.")
                            
                                if os.path.exists(mp3_path):
                                    os.remove(mp3_path)
                            
                        else:
                            yt_status.error(prefix + "Failed to download audio.")
                    except Exception as e:
                        st.error(f"Error processing {video['title']}: {e}")

    # Navigation Switcher for Multiple Results
    if st.session_state.yt_results:
        st.divider()
        st.subheader("View Results")

        # Bulk Download Summaries Logic
        combined_summaries = ""
        for res_id, res_data in st.session_state.yt_results.items():
            combined_summaries += "=" * 50 + "\n"
            combined_summaries += f"VIDEO: {res_data['title']}\n"
            combined_summaries += "=" * 50 + "\n\n"
            combined_summaries += "--- SHORT SUMMARY ---\n"
            combined_summaries += f"{res_data['short']}\n\n"
            combined_summaries += "--- DETAILED SUMMARY ---\n"
            combined_summaries += f"{res_data['detailed']}\n\n"
            combined_summaries += "\n"

        st.download_button(
            label="Download All Summaries",
            data=combined_summaries,
            file_name="all_youtube_summaries.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        result_options = {res['title']: vid_id for vid_id, res in st.session_state.yt_results.items()}
        selected_view_title = st.selectbox(
            "Select processed video to view:", 
            list(result_options.keys()), 
            index=list(result_options.values()).index(st.session_state.current_view_video_id) if st.session_state.current_view_video_id in result_options.values() else 0
        )
        st.session_state.current_view_video_id = result_options[selected_view_title]
        
        res = st.session_state.yt_results[st.session_state.current_view_video_id]
        
        st.subheader(f"Results for: {res['title']}")
        
        st.markdown("### Short Summary (Key Takeaways)")
        st.write(res['short'])
        
        st.markdown("### Detailed Summary")
        st.write(res['detailed'])
        
        with st.expander("Show Coherent Transcript"):
            st.write(res['coherent'])
            
        with st.expander("Show Raw Transcription"):
            st.code(res['raw'], language=None)

# Add download buttons for the outputs
st.sidebar.markdown("---")
st.sidebar.header("Download Results")

if st.session_state.raw_transcription:
    st.sidebar.download_button(
        label="Download Raw Transcription",
        data=st.session_state.raw_transcription,
        file_name="raw_transcription.txt",
        mime="text/plain"
    )

if st.session_state.structured_transcription:
    st.sidebar.download_button(
        label="Download Formatted Interview",
        data=st.session_state.structured_transcription,
        file_name="formatted_interview.txt",
        mime="text/plain"
    )

if st.session_state.structured_summary:
    st.sidebar.download_button(
        label="Download Summary",
        data=st.session_state.structured_summary,
        file_name="interview_summary.txt",
        mime="text/plain"
    )
