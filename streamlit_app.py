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
genai.configure(api_key="AIzaSyDxKnyr6NZ7maXk5bSw9TbTug2E-TXSEz4")

# --------------------------------
# THEME TOGGLE
# --------------------------------
theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
        body {background-color: #0E1117; color: white;}
        </style>
    """, unsafe_allow_html=True)

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
# SAFE MEMORY EXTRACTION
# --------------------------------
def flatten_memory(data):
    texts = []

    for item in data.get("conversations", []):
        if isinstance(item, dict) and "dialogue" in item:
            texts.append(item["dialogue"])

    for item in data.get("chat_examples", []):
        if isinstance(item, dict):
            if "user" in item:
                texts.append(item["user"])
            if "bot" in item:
                texts.append(item["bot"])

    for item in data.get("letters", []):
        if isinstance(item, dict) and "content" in item:
            texts.append(item["content"])

    for quote in data.get("quotes", []):
        texts.append(quote)

    if "love_story" in data:
        for v in data["love_story"].values():
            texts.append(v)

    return texts

memory_texts = flatten_memory(memory_data)

# --------------------------------
# LOAD EMBEDDING MODEL
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

    memory_data["chat_examples"].append({
        "user": user,
        "bot": ai
    })

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
# SIDEBAR
# --------------------------------
st.sidebar.title("🧠 Jabez Control Panel")

voice_on = st.sidebar.checkbox("🔊 Voice Output", value=True)

persona_mode = st.sidebar.radio(
    "Persona Mode",
    ["🧠 Memory Mode", "💬 Casual Talk", "🤍 Emotional Support"]
)

face_mode = st.sidebar.checkbox("🎥 Face-to-Face Mode")

show_emotion = st.sidebar.checkbox("Show Emotion Debug")

# --------------------------------
# DISCLAIMER
# --------------------------------
st.warning("""
⚠️ Research Prototype.
Jabez AI is a synthetic persona for study purposes.
It does NOT represent a real human.
Maintains Ethical AI & Responsible AI boundaries.
""")

# --------------------------------
# MAIN TITLE
# --------------------------------
st.title("🤖 Jabez AI")

if "chat" not in st.session_state:
    st.session_state.chat = []

# --------------------------------
# DISPLAY CHAT (WhatsApp Style)
# --------------------------------
for role, msg in st.session_state.chat:

    if role == "user":
        st.markdown(f"""
        <div style='background-color:#DCF8C6;
        padding:10px;border-radius:15px;
        margin:5px 0;text-align:right'>
        {msg}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style='background-color:#F1F0F0;
        padding:10px;border-radius:15px;
        margin:5px 0;text-align:left'>
        {msg}
        </div>
        """, unsafe_allow_html=True)

# --------------------------------
# INPUT
# --------------------------------
st.markdown("---")
user_input = st.text_input("Talk to Jabez:")

if st.button("Send") and user_input.strip():

    st.session_state.chat.append(("user", user_input))

    context = retrieve_context(user_input)
    context_text = "\n".join(context)

    mode_instruction = ""

    if persona_mode == "🧠 Memory Mode":
        mode_instruction = "Use past memories strongly."
    elif persona_mode == "💬 Casual Talk":
        mode_instruction = "Respond short and casual."
    elif persona_mode == "🤍 Emotional Support":
        mode_instruction = "Respond warmly and supportively."

    prompt = f"""
You are Jabez.
You are an AI persona for academic research.
Never claim to be human.
Avoid emotional dependency.
Maintain healthy AI-human boundaries.

{mode_instruction}

Memory Context:
{context_text}

User: {user_input}
Jabez:
"""

    model = genai.GenerativeModel("models/gemini-2.5-flash")

    with st.spinner("Jabez is thinking..."):
        time.sleep(1)
        response = model.generate_content(prompt)

    ai_text = response.text.strip()
    emotion = detect_emotion(ai_text)

    st.session_state.chat.append(("ai", ai_text))
    save_memory(user_input, ai_text)

    # Emotion Badge
    badge_color = {
        "happy": "green",
        "sad": "blue",
        "neutral": "gray"
    }

    st.markdown(f"""
    <div style='padding:5px;
    border-radius:10px;
    background-color:{badge_color[emotion]};
    color:white;width:150px'>
    Emotion: {emotion.upper()}
    </div>
    """, unsafe_allow_html=True)

    # Face Reaction
    if face_mode:
        if emotion == "happy":
            st.image("https://i.imgur.com/1X6a7LQ.png", width=200)
        elif emotion == "sad":
            st.image("https://i.imgur.com/dJb8F8H.png", width=200)
        else:
            st.image("https://i.imgur.com/8Km9tLL.png", width=200)

    if voice_on:
        audio_file = speak(ai_text, emotion)
        st.audio(audio_file)

    if show_emotion:
        st.write("Detected Emotion:", emotion)
