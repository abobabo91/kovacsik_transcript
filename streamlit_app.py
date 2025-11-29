import streamlit as st
import openai
import anthropic
import google.generativeai as genai
import os
import tempfile

# Function to transcribe audio
def transcribe_audio(audio_file, api_key, provider="openai", model="gpt-4o-transcribe", language="en"):
    try:
        if provider.lower() == "openai":
            client = openai.OpenAI(api_key=api_key)
            # Both whisper-1 and gpt-4o-transcribe use the same endpoint structure
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

# Function to format transcription (using Claude as originally designed, or we could make this flexible too)
# Keeping it Claude-focused as per original code unless specified, but user said "keep logic mostly"
def format_transcription(transcription, claude_api_key):
    if not transcription:
        return ""
    
    claude_prompt = f"""I want a transcript of an interview. 
    I give you the continuous text, and your job is to make it a coherent interview text, without changing any wording. 
    Just change the structure and keep the wording. Here is the text:
{transcription}"""
    
    try:
        claude_client = anthropic.Anthropic(api_key=claude_api_key)
        message = claude_client.messages.create(
            model='claude-sonnet-4-5-20250929', # Updated to latest model
            max_tokens=8000,
            temperature=0,
            system="",
            messages=[{"role": "user", "content": claude_prompt}]
        )
        return message.content[0].text
    except Exception as e:
        st.error(f"Error formatting transcription: {e}")
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
st.set_page_config(page_title="Interview Transcription Tool", layout="wide")

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
if 'summary_prompt' not in st.session_state:
    st.session_state.summary_prompt = """I give you the transcription of an interview. It is a customer discovery call about a company, exploring what they do, their business needs, and their methods.

I want to keep every relevant piece of information that has been said, but ignore everything unimportant.
What I need is a summary of the interview for my notes. 
Go question by question. Write the summary of the question and write the answers into bullet points. Shorten every answer, use keywords when possible.
Use exact wording when it matters. Do not add extra wording, but only what has been said.

This is the transcript:"""

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
        col1, col2 = st.columns(2)
        
        with col1:
            transcription_provider = st.selectbox(
                "Transcription Provider",
                ["OpenAI", "Google"],
                index=0
            )
            
            language_options = {"English": "en", "Hungarian": "hu"}
            selected_language_label = st.selectbox(
                "Language",
                list(language_options.keys()),
                index=0
            )
            selected_language = language_options[selected_language_label]
            
        with col2:
            if transcription_provider == "OpenAI":
                transcription_model = st.selectbox(
                    "OpenAI Model",
                    ["gpt-4o-transcribe", "whisper-1"],
                    index=0,
                    help="gpt-4o-transcribe is generally more accurate than whisper-1."
                )
                t_api_key = openai_api_key
            else:
                transcription_model = st.selectbox(
                    "Google Model",
                    ["gemini-2.5-flash", "gemini-3-pro-preview"],
                    index=0,
                    help="Gemini 2.5 Flash is fast and efficient for audio."
                )
                t_api_key = google_api_key

        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "m4a", "wav"])
        
        if uploaded_file:
            if st.button("Transcribe Audio"):
                if not t_api_key:
                    st.error(f"Missing {transcription_provider} API Key. Please add it to your .streamlit/secrets.toml file.")
                else:
                    with st.spinner(f"Transcribing audio with {transcription_provider} ({transcription_model})..."):
                        raw_transcription = transcribe_audio(uploaded_file, t_api_key, transcription_provider, transcription_model, selected_language)
                        if raw_transcription:
                            st.session_state.raw_transcription = raw_transcription
                            st.success("Transcription complete!")
    
    else:  # "Enter Text Directly"
        raw_transcription_input = st.text_area(
            "Enter your raw transcription text here:",
            height=250,
            value=st.session_state.raw_transcription
        )
        
        if st.button("Use This Text"):
            st.session_state.raw_transcription = raw_transcription_input
            st.success("Text saved as raw transcription!")
    
    # Display the raw transcription if it exists
    if st.session_state.raw_transcription:
        st.subheader("Raw Transcription")
        st.text_area("Raw transcription text", st.session_state.raw_transcription, height=200)
        
        # Format button appears after we have a raw transcription
        if st.button("Format Transcription (Claude)"):
            if not claude_api_key:
                st.error("Missing Claude API Key. Please add it to your .streamlit/secrets.toml file for formatting.")
            else:
                with st.spinner("Formatting transcription with Claude..."):
                    formatted_transcription = format_transcription(st.session_state.raw_transcription, claude_api_key)
                    if formatted_transcription:
                        st.session_state.structured_transcription = formatted_transcription
                        st.success("Formatting complete!")
    
    # Display the formatted transcription if it exists
    if st.session_state.structured_transcription:
        st.subheader("Formatted Interview")
        st.text_area("Formatted interview text", st.session_state.structured_transcription, height=300)


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
            model_provider = st.radio("AI Provider", ["Claude", "OpenAI", "Google"])
        
        with col_mod:
            # Display appropriate model options based on provider
            if model_provider == "Claude":
                # Latest Anthropic Models (Late 2025)
                claude_model = st.selectbox(
                    "Select Claude Model",
                    [
                        "claude-sonnet-4-5-20250929", 
                        "claude-opus-4-5-20251101", 
                        "claude-haiku-4-5-20251001"
                    ],
                    index=0,
                    format_func=lambda x: x.split("-2025")[0].replace("-", " ").title() # Beautify names
                )
                st.session_state.selected_model = claude_model
                api_key = claude_api_key
                provider = "claude"
                
            elif model_provider == "OpenAI":
                # Latest OpenAI Models (Late 2025)
                openai_model = st.selectbox(
                    "Select OpenAI Model",
                    ["gpt-5.1", "gpt-4.1", "gpt-4o", "o3-mini", "o1"],
                    index=0
                )
                st.session_state.selected_model = openai_model
                api_key = openai_api_key
                provider = "openai"
                
            else:  # Google
                # Latest Google Models (Late 2025)
                google_model = st.selectbox(
                    "Select Google Model",
                    ["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash"],
                    index=0
                )
                st.session_state.selected_model = google_model
                api_key = google_api_key
                provider = "google"
        
        # Button to generate/regenerate summary
        if st.button("Generate/Regenerate Summary"):
            if not api_key:
                 st.error(f"Missing {model_provider} API Key. Please add it to your .streamlit/secrets.toml file.")
            else:
                with st.spinner(f"Generating summary with {st.session_state.selected_model}..."):
                    st.session_state.summary_prompt = st.session_state.summary_prompt  # Save the edited prompt
                    st.session_state.structured_summary = summarize_transcription(
                        st.session_state.structured_transcription, 
                        api_key,
                        st.session_state.summary_prompt,
                        provider=provider,
                        model=st.session_state.selected_model
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
