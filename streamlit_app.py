import streamlit as st
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile
from datetime import datetime

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Jabez – Ethical AI Persona", layout="wide")

# ==========================================
# API KEY (PUT YOUR KEY HERE SAFELY)
# ==========================================
genai.configure(api_key="AIzaSyCWButxfN6tbkgbid-vs0mcpK4idrpP0cI")

# ==========================================
# LOAD DATASET
# ==========================================
@st.cache_resource
def load_dataset():
    with open("dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

memory_data = load_dataset()

# ==========================================
# FLATTEN MEMORY SAFELY
# ==========================================
def flatten_memory(data):
    texts = []

    def recurse(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
        else:
            texts.append(str(obj))

    recurse(data)
    return texts

memory_texts = flatten_memory(memory_data)

# ==========================================
# EMBEDDINGS
# ==========================================
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(memory_texts, convert_to_tensor=True)
    return model, embeddings

model, embeddings = load_embeddings()

# ==========================================
# MEMORY RETRIEVAL
# ==========================================
def retrieve_context(query, top_k=5):
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:top_k]
    return [memory_texts[i] for i in top_idx]

# ==========================================
# EMOTION INTENSITY ENGINE
# ==========================================
def emotion_engine(text):
    text = text.lower()
    emotions = {
        "happy": 0,
        "sad": 0,
        "dependency": 0,
        "neutral": 0.2
    }

    happy_words = ["happy", "love", "excited", "great"]
    sad_words = ["sad", "lonely", "cry", "miss"]
    dependency_words = ["only you", "need you", "don't leave", "without you"]

    for w in happy_words:
        if w in text:
            emotions["happy"] += 0.3

    for w in sad_words:
        if w in text:
            emotions["sad"] += 0.4

    for w in dependency_words:
        if w in text:
            emotions["dependency"] += 0.5

    return emotions

# ==========================================
# ETHICAL FILTER
# ==========================================
def ethical_filter(response):
    blocked_phrases = [
        "I am human",
        "I am real",
        "I exist physically"
    ]
    for phrase in blocked_phrases:
        response = response.replace(phrase, "I am an AI system")
    return response

# ==========================================
# TEXT TO SPEECH
# ==========================================
def speak(text):
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# ==========================================
# SIDEBAR CONTROL PANEL
# ==========================================
st.sidebar.title("🧠 Jabez Control Panel")

persona_mode = st.sidebar.radio(
    "Persona Mode",
    ["🧠 Memory Mode", "💬 Casual Talk", "🤍 Emotional Support"]
)

voice_on = st.sidebar.checkbox("🔊 Voice Output", value=True)
show_transparency = st.sidebar.checkbox("🧠 Show Transparency Panel", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Ethical AI • Responsible AI • Human-AI Boundaries")

# ==========================================
# MAIN UI
# ==========================================
st.title("🤖 Jabez – Ethical AI Companion")

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://i.imgur.com/8Km9tLL.png", caption="Jabez AI Persona")

with col2:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f"🧍 **You:** {msg}")
        else:
            st.markdown(f"🤖 **Jabez:** {msg}")

# ==========================================
# USER INPUT
# ==========================================
st.markdown("---")
user_input = st.text_input("Talk to Jabez:")

if st.button("Send"):
    if user_input.strip():

        # Store user message
        st.session_state.chat.append(("user", user_input))

        # Emotion Detection
        emotions = emotion_engine(user_input)

        # Dependency Risk
        dependency_score = emotions["dependency"]

        # Retrieve Memory
        context = retrieve_context(user_input)
        context_text = "\n".join(context)

        # Persona Instructions
        persona_instruction = ""
        if "Memory" in persona_mode:
            persona_instruction = "Answer based strongly on stored memories."
        elif "Casual" in persona_mode:
            persona_instruction = "Answer casually and lightly."
        elif "Emotional" in persona_mode:
            persona_instruction = "Respond warmly but avoid emotional dependency."

        # Prompt
        prompt = f"""
You are Jabez, an ethical AI persona.
You are not human and must never claim to be human.

{persona_instruction}

Memory Context:
{context_text}

User: {user_input}
AI:
"""

        model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model_gemini.generate_content(prompt)

        ai_text = response.text.strip()

        # Ethical filter
        ai_text = ethical_filter(ai_text)

        # Dependency control
        if dependency_score > 0.4:
            ai_text += "\n\n💛 Remember, real-world relationships and self-growth are important too."

        # Save AI message
        st.session_state.chat.append(("ai", ai_text))

        # Voice
        if voice_on:
            audio = speak(ai_text)
            st.audio(audio)

# ==========================================
# TRANSPARENCY PANEL
# ==========================================
if show_transparency and "chat" in st.session_state:
    st.markdown("---")
    st.markdown("### 🧠 Transparency Panel")

    if user_input:
        emotions = emotion_engine(user_input)

        st.write("**Persona Mode:**", persona_mode)
        st.write("**Emotion Scores:**", emotions)
        st.write("**Dependency Risk Level:**",
                 "High" if emotions["dependency"] > 0.4 else "Low")

