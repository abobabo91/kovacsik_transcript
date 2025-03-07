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
    
def summarize_transcription(structured_transcription, api_key, summary_prompt, provider="claude", model=None):
    """
    Summarize transcription using either Claude or OpenAI models.
    
    Args:
        structured_transcription (str): The transcription text to summarize
        api_key (str): API key for the selected provider
        summary_prompt (str): The prompt to guide the summarization
        provider (str): Either "claude" or "openai"
        model (str): Model to use (defaults to appropriate model if None)
    
    Returns:
        str: The generated summary or empty string on failure
    """
    if not structured_transcription:
        return ""
    
    prompt = f"""{summary_prompt}
{structured_transcription}"""
    
    try:
        if provider.lower() == "claude":
            import anthropic
            
            # Default to latest Claude model if none specified
            claude_model = model or "claude-3-7-sonnet-20250219"
            
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
            
            # Default to GPT-4o if none specified
            openai_model = model or "gpt-4o"
            
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=openai_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096  # Adjust as needed for your application
            )
            return response.choices[0].message.content
            
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'claude' or 'openai'.")
            
    except Exception as e:
        import streamlit as st
        st.error(f"Error summarizing transcription with {provider}: {e}")
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

I want to keep every relevant piece of information that has been said, but ignore everything unimportant.
What I need is a summary of the interview for my notes. 
Go question by question. Write the summary of the question and write the answers into bullet points. Shorten every answer, use keywords when possible.
Use exact wording when it matters. Do not add extra wording, but only what has been said.

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
        st.text_area("Raw transcription text", st.session_state.raw_transcription, height=200)
        
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
            height=200
        )
        
        # Model selection options
        st.subheader("Select AI Model")
        model_provider = st.radio("AI Provider", ["Claude", "OpenAI"])
        
        # Initialize model selection in session state if not present
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "claude-3-7-sonnet-20250219" if model_provider == "Claude" else "gpt-4o"
        
        # Display appropriate model options based on provider
        if model_provider == "Claude":
            claude_model = st.selectbox(
                "Select Claude Model",
                ["claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20240620"],
                index=0
            )
            st.session_state.selected_model = claude_model
            api_key = claude_api_key
            provider = "claude"
        else:  # OpenAI
            openai_model = st.selectbox(
                "Select OpenAI Model",
                ["gpt-4o", "gpt-4-turbo"],
                index=0
            )
            st.session_state.selected_model = openai_model
            api_key = openai_api_key  # Make sure you have this defined
            provider = "openai"
        
        # Button to generate/regenerate summary
        if st.button("Generate/Regenerate Summary"):
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