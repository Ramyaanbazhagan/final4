import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="Jabez AI", layout="wide")

# --------------------------------
# LOAD API KEY SECURELY
# --------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("API key not found. Add GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --------------------------------
# DISCLAIMER
# --------------------------------
st.warning("""
⚠️ Jabez AI is a synthetic research persona.
It is NOT a real human.
It does NOT replace real relationships.
Designed under Ethical AI & Responsible AI principles.
""")

# --------------------------------
# LOAD DATASET
# --------------------------------
MEMORY_FILE = "dataset.json"

if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump({"chat_examples": []}, f)

with open(MEMORY_FILE, "r", encoding="utf-8") as f:
    memory_data = json.load(f)

# --------------------------------
# MEMORY FLATTEN
# --------------------------------
def flatten_memory(data):
    texts = []

    for item in data.get("chat_examples", []):
        if isinstance(item, dict):
            if "user" in item:
                texts.append(item["user"])
            if "bot" in item:
                texts.append(item["bot"])

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
    memory_data.setdefault("chat_examples", [])
    memory_data["chat_examples"].append({"user": user, "bot": ai})

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2)

# --------------------------------
# EMOTION DETECTION
# --------------------------------
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["sad", "lonely", "miss", "cry", "hurt"]):
        return "sad"
    if any(w in t for w in ["happy", "love", "excited", "great"]):
        return "happy"
    return "neutral"

# --------------------------------
# VOICE OUTPUT
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

show_emotion = st.sidebar.checkbox("Show Emotion Debug")

# --------------------------------
# MAIN UI
# --------------------------------
st.title("🤖 Jabez AI")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Chat bubble styling
st.markdown("""
<style>
.user-bubble {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
}
.bot-bubble {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
}
</style>
""", unsafe_allow_html=True)

for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='user-bubble'>🧍 <b>You:</b> {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>🤖 <b>Jabez:</b> {msg}</div>", unsafe_allow_html=True)

st.markdown("---")
user_input = st.text_input("Talk to Jabez:")

# --------------------------------
# GENERATION
# --------------------------------
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

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        ai_text = response.text.strip()
    except Exception as e:
        st.error(f"Model generation failed: {e}")
        st.stop()

    emotion = detect_emotion(ai_text)

    st.session_state.chat.append(("ai", ai_text))
    save_memory(user_input, ai_text)

    if voice_on:
        audio_file = speak(ai_text, emotion)
        st.audio(audio_file)

    if show_emotion:
        st.info(f"Emotion Detected: {emotion}")
