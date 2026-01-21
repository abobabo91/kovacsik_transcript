import streamlit as st
import openai
import anthropic
import google.generativeai as genai
import os
import tempfile
from pydub import AudioSegment
from pydub.utils import make_chunks

# Function to transcribe audio
def transcribe_audio(audio_file, api_key, provider="openai", model="gpt-4o-transcribe", language="en", limit_minutes=0):
    try:
        if provider.lower() == "openai":
            client = openai.OpenAI(api_key=api_key)
            
            # Check file size (OpenAI limit is 25MB)
            audio_file.seek(0, os.SEEK_END)
            file_size = audio_file.tell()
            audio_file.seek(0)
            
            limit_bytes = 10 * 1024 * 1024  # 25 MB
            
            # Determine if we need to use pydub (for trimming or splitting)
            use_pydub = (limit_minutes > 0) or (file_size > limit_bytes)
            
            if use_pydub:
                status_msg = "Processing audio..."
                if file_size > limit_bytes:
                    status_msg = f"File size ({file_size / (1024*1024):.2f} MB) exceeds OpenAI 25MB limit."
                if limit_minutes > 0:
                    status_msg += f" Trimming to first {limit_minutes} minutes."
                
                st.warning(status_msg)
                
                full_transcript = ""
                try:
                    # Load audio using pydub
                    audio = AudioSegment.from_file(audio_file)
                    
                    # Trim if needed
                    if limit_minutes > 0:
                        limit_ms = limit_minutes * 60 * 1000
                        audio = audio[:limit_ms]
                    
                    # Check if we still need to split (if duration > 10 mins)
                    # 10 mins of high quality audio is safe for 25MB limit
                    chunk_length_ms = 10 * 60 * 1000
                    
                    if len(audio) > chunk_length_ms:
                        chunks = make_chunks(audio, chunk_length_ms)
                        st.info(f"Audio duration: {len(audio)/60000:.1f} mins. Split into {len(chunks)} chunks for transcription.")
                        
                        progress_bar = st.progress(0)
                        
                        for i, chunk in enumerate(chunks):
                            chunk_temp_path = None
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                                    chunk_temp_path = tmp_file.name
                                
                                # Export chunk (using mp3 high quality to minimize loss)
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
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(chunks))
                    else:
                        # Short enough to send as single file (but processed via pydub)
                        st.info(f"Audio duration: {len(audio)/60000:.1f} mins. Transcribing...")
                        chunk_temp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                                chunk_temp_path = tmp_file.name
                            
                            audio.export(chunk_temp_path, format="mp3", bitrate="192k")
                            
                            with open(chunk_temp_path, "rb") as audio_chunk_file:
                                response = client.audio.transcriptions.create(
                                    model=model,
                                    file=audio_chunk_file,
                                    language=language
                                )
                                full_transcript = response.text
                        finally:
                            if chunk_temp_path and os.path.exists(chunk_temp_path):
                                os.unlink(chunk_temp_path)
                        
                    return full_transcript.strip()
                    
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                    return None
            else:
                # Process normally if small enough and no limit
                response = client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language=language
                )
                return response.text
            
        elif provider.lower() == "google":
            genai.configure(api_key=api_key)
            
            # Create a temporary file to handle the uploaded audio data
            # Gemini Python SDK often prefers file paths or specific blob handling
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(audio_file.getvalue())
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

# Display warning if keys are missing (optional, but helpful for debugging)
if not openai_api_key:
    st.sidebar.warning("OpenAI API Key not found in secrets (`[openai] OPENAI_API_KEY`).")
if not claude_api_key:
    st.sidebar.warning("Claude API Key not found in secrets (`[anthropic] ANTHROPIC_API_KEY`).")
if not google_api_key:
    st.sidebar.warning("Google Gemini API Key not found in secrets (`[google] GEMINI_API_KEY`).")

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

I want to keep every relevant piece of information that has been said, but ignore everything unimportant.
What I need is a summary of the interview for my notes. 
Go question by question. Write the summary of the question and write the answers into bullet points. Shorten every answer, use keywords when possible.
Use exact wording when it matters. Do not add extra wording, but only what has been said.

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

This would be an ideal output, in this exact format:

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
tab1, tab2 = st.tabs(["Transcribe & Format", "Summarize"])

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
