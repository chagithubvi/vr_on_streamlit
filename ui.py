# ui.py

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from voice_recognition import extract_embedding, load_enrolled_embeddings, compute_speaker_thresholds, recognize
from intents import get_speech_input, aayva_response_from_text
import edge_tts
import io
import base64
import asyncio
import os
from langchain_groq import ChatGroq

# Load environment variables
API_KEY = os.getenv("API_KEY")
DEEPGRAM_KEY = os.getenv("DEEPGRAM_KEY")
MODEL = os.getenv("MODEL")

def play_tts(response_text):
    tts_text = response_text.replace("Aiva", "Aayva").replace("Eva", "Aayva")
    async def synthesize_and_return_audio(text):
        communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
        stream = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                stream.write(chunk["data"])
        stream.seek(0)
        return stream
    audio_stream = asyncio.run(synthesize_and_return_audio(tts_text))
    try:
        audio_bytes = audio_stream.read()
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        html_audio = f"""
            <audio autoplay style="display:none;">
                <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
            </audio>
        """
        st.markdown(html_audio, unsafe_allow_html=True)
    finally:
        audio_stream.close()  # Ensure stream is closed to prevent resource leaks

def run_ui():
    # Session state initialization
    if "user_recognized" not in st.session_state:
        st.session_state.user_recognized = False
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "awaiting_tts" not in st.session_state:
        st.session_state.awaiting_tts = False
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = ChatGroq(groq_api_key=API_KEY, model=MODEL)

    # UI setup
    st.markdown("""
        <h1 style='color: #2C3E50; font-weight: 700;'>
            Aiva: Your Smart Home Assistant
        </h1>
    """, unsafe_allow_html=True)

    # Verification
    if not st.session_state.user_recognized:
        verify_audio = audio_recorder(key="verify_recorder")
        if verify_audio:
            test_emb = extract_embedding(verify_audio)
            known_speakers, thresholds = load_enrolled_embeddings()
            speaker_thresholds = compute_speaker_thresholds(known_speakers)
            speaker, _, score = recognize(test_emb, known_speakers, speaker_thresholds)
            if speaker != "Unknown":
                st.session_state.user_recognized = True
                st.success(f"Recognized as: {speaker} (score: {score:.2f})")
                st.session_state.awaiting_tts = True
                st.session_state.conversation_history.append({
                    "user": "Voice verification",
                    "aayva": f"Hello {speaker}, how can I help you?"
                })
                st.rerun()
            else:
                st.error("Voice not recognized. Please try again.")

    # Conversation mode
    else:
        convo_audio = audio_recorder(key="convo_recorder")
        if convo_audio:
            user_input = asyncio.run(get_speech_input(convo_audio))
            if user_input:
                # Replace variations with "Aiva" for display
                display_input = user_input.replace("Eva", "Aiva").replace("Ava", "Aiva").replace("Eiva", "Aiva")
                response, updated_history = aayva_response_from_text(user_input, st.session_state.conversation_history)
                # Update the last entry's user input for display
                updated_history[-1]["user"] = display_input
                st.session_state.conversation_history = updated_history
                st.session_state.awaiting_tts = True
            else:
                st.error("No speech detected. Please try again.")

        # Text input
        def handle_text_input():
            user_input = st.session_state.text_input.strip()
            if user_input:
                # Replace variations with "Aiva" for display
                display_input = user_input.replace("Eva", "Aiva").replace("Ava", "Aiva").replace("Eiva", "Aiva")
                response, updated_history = aayva_response_from_text(user_input, st.session_state.conversation_history)
                # Update the last entry's user input for display
                updated_history[-1]["user"] = display_input
                st.session_state.conversation_history = updated_history
                st.session_state.awaiting_tts = True
                st.session_state.text_input = ""
        st.text_input("Or type here:", key="text_input", on_change=handle_text_input)

        # Display conversation
        for entry in st.session_state.conversation_history:
            col1, col2 = st.columns([0.3, 0.7])
            with col2:
                st.markdown(f"""
                    <div style='text-align: right; background-color: #DCF8C6; color:#1C3F73; padding: 10px 15px; border-radius: 15px; margin: 5px;'>
                        <b>You:</b> {entry['user']}
                    </div>
                """, unsafe_allow_html=True)
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.markdown(f"""
                    <div style='text-align: left; background-color: #E4E6EB; color: #1C3F73; padding: 10px 15px; border-radius: 15px; margin: 5px;'>
                        <b>Aiva:</b> {entry['aayva'] if entry['aayva'] else "..."}
                    </div>
                """, unsafe_allow_html=True)

        # Play TTS for the latest response
        if st.session_state.awaiting_tts and st.session_state.conversation_history:
            last = st.session_state.conversation_history[-1]
            if last["aayva"]:
                play_tts(last["aayva"])
                st.session_state.awaiting_tts = False