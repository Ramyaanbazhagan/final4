import streamlit as st
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile
from datetime import datetime

# ==============================
# CONFIG
# ==============================

st.set_page_config(page_title="Jabez – Ethical AI Persona", layout="wide")

# 🔑 API KEY (Inside file – No secrets error)
genai.configure(api_key="YOUR_GEMINI_API_KEY_HERE")

BOT_NAME = "Jabez"

# ==============================
# SAFE JSON LOAD
# ==============================

def load_memory():
    try:
        with open("dataset.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

memory_data = load_memory()

# ==============================
# FLATTEN MEMORY SAFELY
# ==============================

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

# ==============================
# EMBEDDINGS
# ==============================

@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(memory_texts, convert_to_tensor=True)
    return model, embeddings

model, embeddings = load_model()

# ==============================
# MEMORY SEARCH
# ==============================

def retrieve_context(query, top_k=5):
    if not memory_texts:
        return []
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:top_k]
    return [memory_texts[i] for i in top_idx]

# ==============================
# EMOTION DETECTION
# ==============================

def detect_emotion(text):
    t = text.lower()

    if any(w in t for w in ["sad", "lonely", "cry", "miss"]):
        return "sad"
    if any(w in t for w in ["happy", "excited", "love"]):
        return "happy"
    if any(w in t for w in ["angry", "upset", "fight"]):
        return "tense"
    return "neutral"

# ==============================
# VOICE
# ==============================

def speak(text, emotion):
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# ==============================
# SESSION STATE
# ==============================

if "chat" not in st.session_state:
    st.session_state.chat = []

if "dependency_score" not in st.session_state:
    st.session_state.dependency_score = 0

if "memory_strength" not in st.session_state:
    st.session_state.memory_strength = 0

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("🧠 Jabez Control Panel")

mode = st.sidebar.radio(
    "Persona Mode",
    ["🧠 Memory Mode", "💬 Casual Talk", "🤍 Emotional Support"]
)

voice_on = st.sidebar.checkbox("🔊 Voice Output", value=True)
show_reasoning = st.sidebar.checkbox("🧠 Show Analytics")

emotion_intensity = st.sidebar.slider("Emotion Intensity", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.info("⚠ Ethical AI Prototype\nThis AI does not replace real relationships.")

# ==============================
# MAIN UI
# ==============================

st.title("🤖 Jabez – Ethical Synthetic Persona")

col1, col2 = st.columns([1,2])

with col1:
    st.image("https://i.imgur.com/8Km9tLL.png", use_column_width=True)

with col2:
    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f"🧍 **You:** {msg}")
        else:
            st.markdown(f"🤖 **{BOT_NAME}:** {msg}")

# ==============================
# INPUT
# ==============================

st.markdown("---")
user_input = st.text_input("Talk to Jabez:")

if st.button("Send") and user_input:

    st.session_state.chat.append(("user", user_input))

    # Dependency monitor
    if any(word in user_input.lower() for word in ["only you", "can't live", "need you always"]):
        st.session_state.dependency_score += 1

    context = retrieve_context(user_input)
    context_text = "\n".join(context)

    # Mode control
    if mode == "🤍 Emotional Support":
        style_instruction = "Respond softly, supportive and short."
    elif mode == "💬 Casual Talk":
        style_instruction = "Respond friendly and normal length."
    else:
        style_instruction = "Use memory context strongly in response."

    prompt = f"""
You are {BOT_NAME}, an ethical AI persona.
You never claim to be human.
Avoid emotional dependency.

Emotion Intensity Level: {emotion_intensity}

Mode: {mode}

Memory Context:
{context_text}

User: {user_input}
{BOT_NAME}:
"""

    response = genai.GenerativeModel(
        "models/gemini-1.5-flash"
    ).generate_content(prompt)

    ai_text = response.text.strip()

    emotion = detect_emotion(ai_text)

    st.session_state.chat.append(("ai", ai_text))

    # Save chat into JSON memory
    if "chat_history" not in memory_data:
        memory_data["chat_history"] = []

    memory_data["chat_history"].append({
        "time": str(datetime.now()),
        "user": user_input,
        "bot": ai_text
    })

    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2)

    # Increase memory strength
    st.session_state.memory_strength += 1

    # Voice
    if voice_on:
        audio = speak(ai_text, emotion)
        st.audio(audio)

# ==============================
# ANALYTICS PANEL
# ==============================

if show_reasoning:
    st.markdown("## 🧠 AI Analytics Dashboard")

    st.write("Detected Emotion:", emotion)
    st.write("Memory Strength Score:", st.session_state.memory_strength)
    st.write("Dependency Score:", st.session_state.dependency_score)

    if st.session_state.dependency_score > 3:
        st.warning("⚠ Emotional dependency risk detected.")

    st.write("Persona Mode:", mode)
