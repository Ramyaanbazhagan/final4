import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import time

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="Jabez AI", layout="wide")

# --------------------------------
# API KEY
# --------------------------------
genai.configure(api_key="AIzaSyDnyYEA5ctE-XxsyUp9lYYe4StU_o0_Sd0")

# --------------------------------
# LOAD DATASET
# --------------------------------
MEMORY_FILE = "dataset.json"

if not os.path.exists(MEMORY_FILE):
    st.error("dataset.json file not found.")
    st.stop()

with open(MEMORY_FILE, "r", encoding="utf-8") as f:
    memory_data = json.load(f)

# --------------------------------
# FLATTEN MEMORY
# --------------------------------
def flatten_memory(data):
    texts = []

    for item in data.get("conversations", []):
        if isinstance(item, dict) and "dialogue" in item:
            texts.append(item["dialogue"])

    for item in data.get("chat_examples", []):
        if isinstance(item, dict):
            texts.append(item.get("user", ""))
            texts.append(item.get("bot", ""))

    for item in data.get("letters", []):
        texts.append(item.get("content", ""))

    for quote in data.get("quotes", []):
        texts.append(quote)

    return [t for t in texts if t.strip()]

memory_texts = flatten_memory(memory_data)

# --------------------------------
# EMBEDDINGS
# --------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

def retrieve_context(query, top_k=3):
    if not memory_texts:
        return []

    embeddings = embed_model.encode(memory_texts, convert_to_tensor=True)
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:top_k]
    return [memory_texts[i] for i in top_idx]

# --------------------------------
# SAVE MEMORY
# --------------------------------
def save_memory(user, ai):
    if "chat_examples" not in memory_data:
        memory_data["chat_examples"] = []

    memory_data["chat_examples"].append({"user": user, "bot": ai})

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2)

# --------------------------------
# EMOTION DETECTION
# --------------------------------
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["sad", "miss", "lonely", "cry", "hurt"]):
        return "sad"
    if any(w in t for w in ["happy", "love", "excited", "great"]):
        return "happy"
    return "neutral"

# --------------------------------
# VOICE
# --------------------------------
def speak(text, emotion):
    slow = True if emotion == "sad" else False
    tts = gTTS(text, slow=slow)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# --------------------------------
# SIDEBAR CONTROLS
# --------------------------------
st.sidebar.title("🧠 Jabez Control Panel")

voice_on = st.sidebar.checkbox("🔊 Voice Output", value=True)

persona_mode = st.sidebar.radio(
    "Persona Mode",
    ["🧠 Memory Mode", "💬 Casual Talk", "🤍 Emotional Support"]
)

face_mode = st.sidebar.checkbox("🎥 Face-to-Face Mode")

theme_mode = st.sidebar.radio("Theme", ["Light", "Dark"])

# --------------------------------
# THEME CSS
# --------------------------------
if theme_mode == "Dark":
    st.markdown("""
    <style>
    body {background-color:#0e1117;color:white;}
    .stTextInput>div>div>input {background-color:#1e2228;color:white;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    body {background-color:#f5f7fa;}
    </style>
    """, unsafe_allow_html=True)

# --------------------------------
# DISCLAIMER
# --------------------------------
st.warning("""
⚠️ Research Prototype – Synthetic AI Persona.
Does NOT represent a real human.
Maintains ethical AI-human boundaries.
""")

# --------------------------------
# MAIN TITLE
# --------------------------------
st.title("🤖 Jabez AI Neural Persona")

if "chat" not in st.session_state:
    st.session_state.chat = []

# --------------------------------
# CHAT BUBBLES
# --------------------------------
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"""
        <div style='background:#DCF8C6;padding:10px;border-radius:10px;margin:5px;text-align:right;'>
        🧍 {msg}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#E4E6EB;padding:10px;border-radius:10px;margin:5px;text-align:left;'>
        🤖 {msg}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
user_input = st.text_input("Talk to Jabez:")

# --------------------------------
# GENERATE RESPONSE
# --------------------------------
if st.button("Send") and user_input.strip():

    st.session_state.chat.append(("user", user_input))

    with st.spinner("Jabez is thinking..."):
        time.sleep(1)

        context = retrieve_context(user_input)
        context_text = "\n".join(context)

        mode_instruction = ""
        if persona_mode == "🧠 Memory Mode":
            mode_instruction = "Use memory context strongly."
        elif persona_mode == "💬 Casual Talk":
            mode_instruction = "Respond short and casual."
        else:
            mode_instruction = "Respond warmly and supportively."

        prompt = f"""
You are Jabez.
You are a synthetic AI persona for academic research.
Never claim to be human.
Avoid emotional dependency.

{mode_instruction}

Memory Context:
{context_text}

User: {user_input}
Jabez:
"""

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        ai_text = response.text.strip()

    emotion = detect_emotion(ai_text)

    st.session_state.chat.append(("ai", ai_text))
    save_memory(user_input, ai_text)

    # --------------------------------
    # EMOTION AVATAR
    # --------------------------------
    if face_mode:
        if emotion == "happy":
            st.image("https://i.imgur.com/1XqQZ5F.png", width=200)
        elif emotion == "sad":
            st.image("https://i.imgur.com/Q6aZQ4S.png", width=200)
        else:
            st.image("https://i.imgur.com/8Km9tLL.png", width=200)

    # Emotion Badge
    color = "green" if emotion=="happy" else "blue" if emotion=="neutral" else "red"
    st.markdown(f"""
    <span style='background:{color};color:white;padding:5px 10px;border-radius:10px;'>
    Emotion: {emotion.upper()}
    </span>
    """, unsafe_allow_html=True)

    # Voice
    if voice_on:
        audio_file = speak(ai_text, emotion)
        st.audio(audio_file)
