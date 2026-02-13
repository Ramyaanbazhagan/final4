import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import time

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Jabez AI", layout="wide")

# -------------------------------
# API KEY (Use Streamlit Secrets)
# -------------------------------
genai.configure(api_key=st.secrets["AIzaSyCI4uglizm35hDx5COaa9uuXMJjdg-VNZg"])

# -------------------------------
# MEMORY FILE
# -------------------------------
MEMORY_FILE = "dataset.json"

if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump({"chat_examples": []}, f)

with open(MEMORY_FILE, "r") as f:
    memory_data = json.load(f)

# -------------------------------
# MEMORY FLATTEN
# -------------------------------
def flatten_memory(data):
    texts = []
    for item in data.get("chat_examples", []):
        if "user" in item:
            texts.append(item["user"])
        if "bot" in item:
            texts.append(item["bot"])
    return texts

memory_texts = flatten_memory(memory_data)

# -------------------------------
# EMBEDDING MODEL
# -------------------------------
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

# -------------------------------
# SAVE MEMORY
# -------------------------------
def save_memory(user, ai):
    memory_data["chat_examples"].append({
        "user": user,
        "bot": ai
    })
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory_data, f, indent=2)

# -------------------------------
# EMOTION DETECTION
# -------------------------------
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["sad", "miss", "lonely", "cry", "hurt"]):
        return "sad"
    if any(w in t for w in ["happy", "love", "excited", "great"]):
        return "happy"
    return "neutral"

# -------------------------------
# VOICE
# -------------------------------
def speak(text, emotion):
    slow = True if emotion == "sad" else False
    tts = gTTS(text, slow=slow)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.title("🧠 Jabez Control Panel")

voice_on = st.sidebar.checkbox("🔊 Voice Output", value=True)

persona_mode = st.sidebar.radio(
    "Persona Mode",
    ["🧠 Memory Mode", "💬 Casual Talk", "🤍 Emotional Support"]
)

dark_mode = st.sidebar.toggle("🌙 Dark Mode")

face_mode = st.sidebar.toggle("👁 Face-to-Face Mode")

# -------------------------------
# THEME
# -------------------------------
if dark_mode:
    bg_color = "#0e1117"
    text_color = "white"
else:
    bg_color = "white"
    text_color = "black"

st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
}}
.chat-bubble-user {{
    background-color: #DCF8C6;
    padding:10px;
    border-radius:10px;
    margin:5px;
    text-align:right;
}}
.chat-bubble-ai {{
    background-color: #E6E6E6;
    padding:10px;
    border-radius:10px;
    margin:5px;
    text-align:left;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# DISCLAIMER
# -------------------------------
st.warning("""
⚠️ Research Prototype.
Jabez AI is a synthetic persona.
It does NOT represent a human.
It does NOT replace real relationships.
Built under Ethical & Responsible AI principles.
""")

# -------------------------------
# TITLE
# -------------------------------
st.title("🤖 Jabez AI")

if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------------------
# DISPLAY CHAT
# -------------------------------
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='chat-bubble-user'>🧍 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-ai'>🤖 {msg}</div>", unsafe_allow_html=True)

# -------------------------------
# INPUT
# -------------------------------
st.markdown("---")
user_input = st.text_input("Talk to Jabez:")

if st.button("Send") and user_input.strip():

    st.session_state.chat.append(("user", user_input))

    with st.spinner("Jabez is typing..."):
        time.sleep(1)

        context = retrieve_context(user_input)
        context_text = "\n".join(context)

        mode_instruction = ""
        if persona_mode == "🧠 Memory Mode":
            mode_instruction = "Use past memory strongly."
        elif persona_mode == "💬 Casual Talk":
            mode_instruction = "Respond short and casual."
        elif persona_mode == "🤍 Emotional Support":
            mode_instruction = "Respond warmly and supportive."

        prompt = f"""
You are Jabez.
You are an AI research persona.
Never claim to be human.
Avoid emotional dependency.
Maintain AI-human boundaries.

{mode_instruction}

Memory:
{context_text}

User: {user_input}
Jabez:
"""

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            ai_text = response.text.strip()
        except Exception:
            st.error("Model generation failed. Check API key or model access.")
            st.stop()

    emotion = detect_emotion(ai_text)

    # Emotion Avatar
    if face_mode:
        if emotion == "happy":
            st.image("https://i.imgur.com/1X8aYqR.png", width=120)
        elif emotion == "sad":
            st.image("https://i.imgur.com/0T9G9kG.png", width=120)
        else:
            st.image("https://i.imgur.com/9bK0K6T.png", width=120)

    st.session_state.chat.append(("ai", ai_text))
    save_memory(user_input, ai_text)

    # Emotion Badge
    st.markdown(f"**Emotion:** `{emotion.upper()}`")

    if voice_on:
        audio_file = speak(ai_text, emotion)
        st.audio(audio_file)

    st.rerun()
