import streamlit as st
import openai
import anthropic
import os

def transcribe_audio(audio_file):
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        response = client.audio.transcriptions.create(
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
        st.warning("Please enter your OpenAI API key to enable transcription.")

if 'claude' in st.secrets and 'CLAUDE_API_KEY' in st.secrets['claude']:
    claude_api_key = st.secrets['claude']["CLAUDE_API_KEY"]
else:
    claude_api_key = st.sidebar.text_input("Claude API Key", type="password")
    if not claude_api_key:
        st.warning("Please enter your Claude API key to enable formatting and summarization.")

# Initialize session state variables
if 'transcription' not in st.session_state:
    st.session_state.transcription = ''
if 'structured_transcription' not in st.session_state:
    st.session_state.structured_transcription = ''
if 'structured_summary' not in st.session_state:
    st.session_state.structured_summary = ''
if 'summary_prompt' not in st.session_state:
    st.session_state.summary_prompt = """I give you the transcription of an interview. It is a customer discovery call about a company,
exploring what they do, their business needs, and their methods.

What I need is a structured summary of the interview for my notes. I want to keep every relevant piece of information that
has been said, but ignore everything unimportant.
Write the answers into bullet points and use exact wording when it matters. Do not add extra wording, but only what has been said.
This is the transcript:"""

st.title("Interview Transcription and Summarization")

# Create tabs for a better UI experience
tab1, tab2, tab3 = st.tabs(["Transcribe", "Format", "Summarize"])

with tab1:
    st.header("Transcribe Audio")
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "m4a", "wav"])
    
    if uploaded_file:
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(uploaded_file)
                if transcription:
                    st.session_state.transcription = transcription
                    st.success("Transcription complete!")
    
    if st.session_state.transcription:
        st.subheader("Raw Transcription")
        st.text_area("Raw transcription text", st.session_state.transcription, height=300)

with tab2:
    st.header("Format Transcription")
    
    if not st.session_state.transcription:
        st.info("Please complete the transcription step first.")
    else:
        if st.button("Format Transcription"):
            with st.spinner("Formatting transcription..."):
                structured_transcription = format_transcription(st.session_state.transcription, claude_api_key)
                if structured_transcription:
                    st.session_state.structured_transcription = structured_transcription
                    st.success("Formatting complete!")
        
        if st.session_state.structured_transcription:
            st.subheader("Formatted Interview")
            st.text_area("Formatted interview text", st.session_state.structured_transcription, height=400)

with tab3:
    st.header("Summarize Interview")
    
    if not st.session_state.structured_transcription:
        st.info("Please complete the formatting step first.")
    else:
        st.subheader("Customize Summary Prompt")
        
        # Let user edit the summary prompt
        summary_prompt = st.text_area(
            "Edit the summary prompt below",
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

if st.session_state.transcription:
    st.sidebar.download_button(
        label="Download Raw Transcription",
        data=st.session_state.transcription,
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