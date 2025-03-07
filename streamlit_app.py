import streamlit as st
import openai
import anthropic
import os

def transcribe_audio(audio_file, openai_api_key):
    try:
        openai_client = openai.OpenAI(api_key=openai_api_key)
        response = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return response.text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

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
            model='claude-3-7-sonnet-20250219',
            max_tokens=8000,
            temperature=0,
            system="",
            messages=[{"role": "user", "content": claude_prompt}]
        )
        return message.content[0].text
    except Exception as e:
        st.error(f"Error formatting transcription: {e}")
        return ""

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
        st.warning("Please enter your OpenAI API key to enable audio transcription.")

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
    st.session_state.summary_prompt = """I give you the transcription of an interview. It is a customer discovery call about a company, exploring what they do, their business needs, and their methods.

What I need is a structured summary of the interview for my notes. I want to keep every relevant piece of information that
has been said, but ignore everything unimportant.
Go question by question and write the answers into bullet points and use exact wording when it matters. Do not add extra wording, but only what has been said.

This is the transcript:"""

st.title("Interview Transcription and Summarization")

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
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "m4a", "wav"])
        
        if uploaded_file:
            if st.button("Transcribe Audio"):
                with st.spinner("Transcribing audio..."):
                    raw_transcription = transcribe_audio(uploaded_file, openai_api_key)
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
        st.text_area("Raw transcription text", st.session_state.raw_transcription, height=200, disabled=True)
        
        # Format button appears after we have a raw transcription
        if st.button("Format Transcription"):
            with st.spinner("Formatting transcription..."):
                formatted_transcription = format_transcription(st.session_state.raw_transcription, claude_api_key)
                if formatted_transcription:
                    st.session_state.structured_transcription = formatted_transcription
                    st.success("Formatting complete!")
    
    # Display the formatted transcription if it exists
    if st.session_state.structured_transcription:
        st.subheader("Formatted Interview")
        st.text_area("Formatted interview text", st.session_state.structured_transcription, height=300, disabled=True)

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
            height=200
        )
        
        # Button to generate/regenerate summary
        if st.button("Generate/Regenerate Summary"):
            with st.spinner("Generating summary..."):
                st.session_state.summary_prompt = st.session_state.summary_prompt  # Save the edited prompt
                st.session_state.structured_summary = summarize_transcription(
                    st.session_state.structured_transcription, 
                    claude_api_key,
                    st.session_state.summary_prompt
                )
                if st.session_state.structured_summary:
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