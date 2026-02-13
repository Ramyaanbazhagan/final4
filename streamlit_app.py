import streamlit as st
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="🧠 Jabez", layout="wide")

# ==================================================
# GEMINI CONFIG (SAFE)
# ==================================================
GEMINI_KEY = "AIzaSyAGzo7IsyufhNzqff-YBlhlLrGfxD4tGcY"
genai.configure(api_key=GEMINI_KEY)

model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# ==================================================
# LOAD MEMORY DATASET
# ==================================================
@st.cache_resource
def load_memory():
    with open("dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

memory_data = load_memory()

# ==================================================
# FLATTEN MEMORY
# ==================================================
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

# ==================================================
# EMBEDDING MODEL
# ==================================================
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(memory_texts, convert_to_tensor=True)
    return model, embeddings

embed_model, embeddings = load_embeddings()

# ==================================================
# MEMORY RETRIEVAL
# ==================================================
def retrieve_context(query, top_k=5):
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:top_k]
    return [memory_texts[i] for i in top_idx]

# ==================================================
# EMOTION DETECTION
# ==================================================
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["sad", "miss", "lonely", "cry"]):
        return "sad"
    if any(w in t for w in ["happy", "excited", "love"]):
        return "happy"
    return "neutral"

# ==================================================
# TEXT TO SPEECH
# ==================================================
def speak(text, slow=False):
    tts = gTTS(text, slow=slow)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# ==================================================
# SAVE CHAT TO MEMORY FILE
# ==================================================
def save_chat(user, bot):
    if "chat_log" not in memory_data:
        memory_data["chat_log"] = []

    memory_data["chat_log"].append({
        "user": user,
        "bot": bot
    })

    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2, ensure_ascii=False)

# ==================================================
# SIDEBAR CONTROL PANEL
# ==================================================
st.sidebar.title("🧠 Jabez Control Panel")

persona_mode = st.sidebar.radio(
    "Persona Mode",
    ["🧠 Memory Mode", "💬 Casual Talk", "🤍 Emotional Support"]
)

voice_on = st.sidebar.checkbox("🔊 Voice Output", value=True)
show_reasoning = st.sidebar.checkbox("🧠 Show Reasoning")

st.sidebar.markdown("---")
st.sidebar.caption("Ethical AI • Responsible AI • Human-AI Boundaries")

# ==================================================
# MAIN UI
# ==================================================
st.title("🤖 Jabez – AI Companion")

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://i.imgur.com/8Km9tLL.png", use_column_width=True)

with col2:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f"🧍 **You:** {msg}")
        else:
            st.markdown(f"🤖 **Jabez:** {msg}")

# ==================================================
# INPUT
# ==================================================
st.markdown("---")
user_input = st.text_input("Talk to Jabez:")

if st.button("Send"):
    if user_input.strip():

        st.session_state.chat.append(("user", user_input))

        context = retrieve_context(user_input)
        context_text = "\n".join(context)

        # Persona behavior control
        if persona_mode == "🧠 Memory Mode":
            instruction = "Use stored memory context deeply."
        elif persona_mode == "💬 Casual Talk":
            instruction = "Respond casually and briefly."
        else:
            instruction = "Respond emotionally supportive and caring."

        prompt = f"""
You are Jabez, a synthetic AI persona.
Never claim to be a real human.
Maintain healthy emotional boundaries.

{instruction}

Memory Context:
{context_text}

User: {user_input}
Jabez:
"""

        try:
            response = model_gemini.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 400
                }
            )

            ai_text = response.text if response.text else "I'm here with you."

        except Exception:
            ai_text = "Sorry, I'm having trouble responding right now."

        emotion = detect_emotion(ai_text)

        # Emotion controls voice speed
        slow_voice = True if emotion == "sad" else False

        st.session_state.chat.append(("ai", ai_text))

        save_chat(user_input, ai_text)

        if voice_on:
            audio_file = speak(ai_text, slow=slow_voice)
            st.audio(audio_file)

        if show_reasoning:
            st.markdown("### 🧠 System Analysis")
            st.write("Emotion detected:", emotion)
            st.write("Persona Mode:", persona_mode)
            st.write("Memory Used:")
            for c in context:
                st.write("-", c)

# ==================================================
# DISCLAIMER
# ==================================================
st.markdown("---")
st.caption(
    "This AI persona is a research prototype. "
    "It does not replace real human relationships."
)
