import streamlit as st
import openai
import anthropic
import os

def transcribe_and_format(audio_file, openai_api_key, claude_api_key):
    # Step 1: Transcribe audio
    try:
        openai_client = openai.OpenAI(api_key=openai_api_key)
        response = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        raw_transcription = response.text
        
        # Step 2: Format transcription
        claude_prompt = f"""I want a transcript of an interview. 
        I give you the continuous text, and your job is to make it a coherent interview text, without changing any wording. 
        Just change the structure and keep the wording. Here is the text:
{raw_transcription}"""
        
        claude_client = anthropic.Anthropic(api_key=claude_api_key)
        message = claude_client.messages.create(
            model='claude-3-7-sonnet-20250219',
            max_tokens=8000,
            temperature=0,
            system="",
            messages=[{"role": "user", "content": claude_prompt}]
        )
        formatted_transcription = message.content[0].text
        
        return raw_transcription, formatted_transcription
    except Exception as e:
        st.error(f"Error in transcription or formatting: {e}")
        return None, None

def summarize_transcription(structured_transcription, claude_api_key, summary_prompt):
    if not structured_transcription:
        return ""
    
    claude_prompt = f"""{summary_prompt}
{structured_transcription}"""
    
    try:
        client = anthropic.Anthropic(api_key=claude_api_key)
        message = client.messages.create(
            model='claude-3-7-sonnet-20250219',
            max_tokens=8000,
            temperature=0,
            system="",
            messages=[{"role": "user", "content": claude_prompt}]
        )
        return message.content[0].text
    except Exception as e:
        st.error(f"Error summarizing transcription: {e}")
        return ""

# Page config for better appearance
st.set_page_config(page_title="Interview Transcription Tool", layout="wide")

# API Keys
if 'openai' in st.secrets and 'OPENAI_API_KEY' in st.secrets['openai']:
    openai_api_key = st.secrets['openai']["OPENAI_API_KEY"]
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to enable transcription.")

if 'claude' in st.secrets and 'CLAUDE_API_KEY' in st.secrets['claude']:
    claude_api_key = st.secrets['claude']["CLAUDE_API_KEY"]
else:
    claude_api_key = st.sidebar.text_input("Claude API Key", type="password")
    if not claude_api_key:
        st.warning("Please enter your Claude API key to enable formatting and summarization.")

# Initialize session state variables
if 'raw_transcription' not in st.session_state:
    st.session_state.raw_transcription = ''
if 'structured_transcription' not in st.session_state:
    st.session_state.structured_transcription = ''
if 'structured_summary' not in st.session_state:
    st.session_state.structured_summary = ''
if 'summary_prompt' not in st.session_state:
    st.session_state.summary_prompt = """I have an interview transcription from a customer discovery call. Please help me create a structured summary that:

Captures all relevant business information (company details, needs, methods, pain points)
Organizes answers by question in bullet point format
Uses direct quotes when the exact wording is significant
Omits filler content, pleasantries, and irrelevant tangents
Maintains the original meaning without adding interpretation

Please provide the transcription, and I'll create a concise, organized summary that preserves all important information while eliminating unnecessary content.

This is the transcript:"""

st.title("Interview Transcription and Summarization")

# Create tabs for a better UI experience
tab1, tab2 = st.tabs(["Transcribe & Format", "Summarize"])

with tab1:
    st.header("Transcribe and Format Audio")
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "m4a", "wav"])
    
    if uploaded_file:
        if st.button("Transcribe & Format Audio"):
            with st.spinner("Transcribing and formatting audio..."):
                raw_transcription, formatted_transcription = transcribe_and_format(uploaded_file, openai_api_key, claude_api_key)
                if raw_transcription and formatted_transcription:
                    st.session_state.raw_transcription = raw_transcription
                    st.session_state.structured_transcription = formatted_transcription
                    st.success("Transcription and formatting complete!")
    
    if st.session_state.raw_transcription and st.session_state.structured_transcription:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Raw Transcription")
            st.text_area("Raw transcription text", st.session_state.raw_transcription, height=300)
        
        with col2:
            st.subheader("Formatted Interview")
            st.text_area("Formatted interview text", st.session_state.structured_transcription, height=300)

with tab2:
    st.header("Summarize Interview")
    
    if not st.session_state.structured_transcription:
        st.info("Please complete the transcription step first.")
    else:
        st.subheader("Customize Summary Prompt")
        
        # Let user edit the summary prompt
        summary_prompt = st.text_area(
            "Edit the summary prompt below. Hit Ctrl + Enter to save.",
            value=st.session_state.summary_prompt,
            height=200
        )
        
        # Button to generate/regenerate summary
        if st.button("Generate/Regenerate Summary"):
            with st.spinner("Generating summary..."):
                st.session_state.summary_prompt = summary_prompt  # Save the edited prompt
                structured_summary = summarize_transcription(
                    st.session_state.structured_transcription, 
                    claude_api_key,
                    summary_prompt
                )
                if structured_summary:
                    st.session_state.structured_summary = structured_summary
                    st.success("Summary generated!")
        
        # Display the summary
        if st.session_state.structured_summary:
            st.subheader("Interview Summary")
            st.text_area("Summary", st.session_state.structured_summary, height=400)

# Add download buttons for the outputs
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