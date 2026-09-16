
import streamlit as st
from google import genai

from rag import DebateRAG
from debate_engine import debate_with_rag
from evaluation import evaluate_debate
from speech import load_speech_model, transcribe_audio, text_to_speech


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Debate Partner",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None


# --------------------------------------------------
# API CONNECTION
# --------------------------------------------------

api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def initialize_rag():

    return DebateRAG(
        "data/knowledge/debate_knowledge.txt"
    )


@st.cache_resource
def initialize_whisper():

    return load_speech_model()


rag = initialize_rag()
whisper_model = initialize_whisper()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🧠 AI Debate Partner")

st.markdown(
    """
### Think. Argue. Defend your position.

Speak or type your argument and challenge yourself against an
AI that takes the **opposing side** of the debate.
"""
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("ℹ️ How it works")

    st.markdown(
        """
        **1. 🎤 Speak or type**

        Give your argument.

        **2. 📝 Speech Recognition**

        Voice input is converted into text using Whisper.

        **3. 📚 RAG**

        Relevant knowledge is retrieved from the knowledge base.

        **4. 🤖 Deep Learning**

        Model analyzes the argument and generates a
        counterargument.

        **5. 🔊 Voice Response**

        The AI response can be played as audio.

        **6. 📊 Evaluation**

        At the end, the AI evaluates your debate performance.
        """
    )

    st.divider()

    st.caption(
        "AI Debate Partner • Voice + Deep Learning + RAG"
    )


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("🎤 Make Your Argument")

input_mode = st.radio(
    "Choose your input method:",
    ["🎤 Voice", "⌨️ Text"],
    horizontal=True
)


user_argument = ""


# --------------------------------------------------
# VOICE INPUT
# --------------------------------------------------

if input_mode == "🎤 Voice":

    audio = st.audio_input(
        "Record your argument"
    )

    if audio is not None:

        st.audio(audio)

        if st.button(
            "📝 Convert Speech to Text",
            use_container_width=True
        ):

            with st.spinner(
                "Converting speech to text..."
            ):

                user_argument = transcribe_audio(
                    whisper_model,
                    audio
                )

            st.session_state["voice_text"] = user_argument


    if "voice_text" in st.session_state:

        user_argument = st.session_state["voice_text"]

        st.success("Speech recognized!")

        st.text_area(
            "📝 Recognized Speech",
            value=user_argument,
            height=120
        )


# --------------------------------------------------
# TEXT INPUT
# --------------------------------------------------

else:

    user_argument = st.text_area(
        "⌨️ Type your argument",
        placeholder=(
            "Example: AI should replace human teachers "
            "because..."
        ),
        height=150
    )


# --------------------------------------------------
# SUBMIT ARGUMENT
# --------------------------------------------------

if st.button(
    "⚔️ Submit Argument",
    type="primary",
    use_container_width=True
):

    if not user_argument.strip():

        st.warning(
            "Please provide an argument first."
        )

    else:

        with st.spinner(
            "🤖 AI is analyzing your argument..."
        ):

            reply, retrieved = debate_with_rag(
                client,
                rag.retrieve,
                user_argument,
                st.session_state.history
            )

        # Save conversation

        st.session_state.history.append(
            {
                "role": "User",
                "content": user_argument
            }
        )

        st.session_state.history.append(
            {
                "role": "AI",
                "content": reply
            }
        )

        # Generate AI audio

        with st.spinner(
            "🔊 Generating voice response..."
        ):

            audio_file = text_to_speech(reply)

        st.session_state.last_audio = audio_file

        st.session_state.evaluation = None


# --------------------------------------------------
# CURRENT DEBATE
# --------------------------------------------------

if st.session_state.history:

    st.divider()

    st.subheader("💬 Debate")

    for message in st.session_state.history:

        if message["role"] == "User":

            with st.chat_message("user"):

                st.markdown("### 📝 You")

                st.write(
                    message["content"]
                )

        else:

            with st.chat_message("assistant"):

                st.markdown(
                    "### 🤖 AI Debate Partner"
                )

                st.write(
                    message["content"]
                )


# --------------------------------------------------
# AUDIO RESPONSE
# --------------------------------------------------

if st.session_state.last_audio:

    st.subheader("🔊 AI Voice Response")

    st.audio(
        st.session_state.last_audio,
        format="audio/mp3"
    )


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

if st.session_state.history:

    st.divider()

    st.subheader("📊 Debate Evaluation")

    if st.button(
        "📊 Evaluate My Debate",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing your debate performance..."
        ):

            evaluation = evaluate_debate(
                client,
                st.session_state.history
            )

        st.session_state.evaluation = evaluation

    if st.session_state.evaluation:

        st.markdown(
            st.session_state.evaluation
        )


# --------------------------------------------------
# NEW DEBATE
# --------------------------------------------------

if st.session_state.history:

    st.divider()

    if st.button(
        "🔄 Start New Debate",
        use_container_width=True
    ):

        st.session_state.history = []

        st.session_state.last_audio = None

        st.session_state.evaluation = None

        if "voice_text" in st.session_state:
            del st.session_state["voice_text"]

        st.rerun()
