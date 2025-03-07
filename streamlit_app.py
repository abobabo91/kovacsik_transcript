import streamlit as st
import openai
import anthropic
import os

def transcribe_audio(audio_file):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return response.text

def format_transcription(transcription, claude_api_key):
    claude_prompt = f"""I want a transcript of an interview. 
    I give you the continuous text, and your job is to make it a coherent interview text, without changing any wording. 
    Just change the structure and keep the wording. Here is the text:

{transcription}"""
    
    client = anthropic.Anthropic(api_key=claude_api_key)
    message = client.messages.create(
        model='claude-3-7-sonnet-20250219',
        max_tokens=8000,
        temperature=0,
        system="",
        messages=[{"role": "user", "content": [{"type": "text", "text": claude_prompt}]}]
    )
    return message.content[0].text

def summarize_transcription(structured_transcription, claude_api_key):
    claude_prompt = f"""I give you the transcription of an interview. It is a customer discovery call about a company,
    exploring what they do, their business needs, and their methods.
    
    What I need is a structured summary of the interview for my notes. I want to keep every relevant piece of information that
    has been said, but ignore everything unimportant.
    Write the answers into bullet points and use exact wording when it matters. Do not add extra wording, but only what has been said.

    This is the transcript:

{structured_transcription}"""
    
    client = anthropic.Anthropic(api_key=claude_api_key)
    message = client.messages.create(
        model='claude-3-7-sonnet-20250219',
        max_tokens=8000,
        temperature=0,
        system="",
        messages=[{"role": "user", "content": [{"type": "text", "text": claude_prompt}]}]
    )
    return message.content[0].text

st.title("Interview Transcription and Summarization")

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
claude_api_key = st.text_input("Enter your Claude API Key", type="password")
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "m4a", "wav"])

if uploaded_file and openai_api_key and claude_api_key:
    st.write("### Transcribing Audio...")
    transcription = transcribe_audio(uploaded_file)
    st.text_area("Raw Transcription", transcription, height=200)
    
    st.write("### Formatting Transcription...")
    structured_transcription = format_transcription(transcription, claude_api_key)
    st.text_area("Formatted Interview", structured_transcription, height=300)
    
    st.write("### Generating Summary...")
    structured_summary = summarize_transcription(structured_transcription, claude_api_key)
    st.text_area("Interview Summary", structured_summary, height=300)
