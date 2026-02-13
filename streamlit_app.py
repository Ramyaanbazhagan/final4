import streamlit as st
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile
from datetime import datetime

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="🧠 Jabez – Neural Persona", layout="wide")

# ==============================
# GEMINI CONFIG (PUT YOUR KEY HERE)
# ==============================
genai.configure(api_key="AIzaSyCWButxfN6tbkgbid-vs0mcpK4idrpP0cI")

# ==============================
# LOAD DATASET
# ==============================
@st.cache_resource
def load_memory():
    with open("dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

memory_data = load_memory()

# ==============================
# FLATTEN MEMORY (SAFE FOR YOUR DATASET)
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
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(memory_texts, convert_to_tensor=True)
    return model, embeddings

model_embed, embeddings = load_embeddings()

# ==============================
# MEMORY RETRIEVAL
# ==============================
def retrieve_context(query, top_k=5):
    query_emb = model_embed.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:top_k]
    return [memory_texts[i] for i in top_idx]

# ==============================
# EMOTION DETECTION
# ==============================
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["sad", "miss", "lonely", "cry", "hurt"]):
        return "sad"
    if any(w in t for w in ["happy", "love", "excited", "great"]):
        return "happy"
    return "neutral"

# ==============================
# TEXT TO SPEECH
# ==============================
def speak(text, slow=False):
    tts = gTTS(text, slow=slow)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# ==============================
# SAVE LONG TERM MEMORY
# ==============================
def save_chat(user, ai):
    entry = {
        "time": str(datetime.now()),
        "user": user,
        "ai": ai
    }

    if "chat_history" not in memory_data:
        memory_data["chat_history"] = []

    memory_data["chat_history"].append(entry)

    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2)

# ==============================
# SIDEBAR CONTROL PANEL
# ==============================
st.sidebar.title("🧠 Jabez Control Panel")

persona_mode = st.sidebar.radio(
    "Persona Mode",
    ["🧠 Memory Mode", "💬 Casual Talk", "🤍 Emotional Support"]
)

voice_on = st.sidebar.checkbox("🔊 Voice Output", value=True)
show_reasoning = st.sidebar.checkbox("🧠 Show Reasoning")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Research Tags")
st.sidebar.caption("""
Ethical AI  
Responsible AI  
Human-AI Boundaries  
Synthetic Persona Modeling
""")

st.sidebar.markdown("---")
st.sidebar.caption("""
⚠️ Disclaimer:
This is a synthetic AI persona built for academic research.
It does not represent a real human identity.
""")

# ==============================
# MAIN UI
# ==============================
st.title("🤖 Jabez – AI Neural Persona")

col1, col2 = st.columns([1, 2])

with col1:
    st.image(
        "https://i.imgur.com/8Km9tLL.png",
        caption="Jabez – Synthetic Persona",
        use_column_width=True
    )

with col2:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f"🧍 **You:** {msg}")
        else:
            st.markdown(f"🤖 **Jabez:** {msg}")

# ==============================
# INPUT
# ==============================
st.markdown("---")
user_input = st.text_input("Talk to Jabez:")

if st.button("Send"):
    if user_input.strip():

        st.session_state.chat.append(("user", user_input))

        # Persona Mode Behavior
        if persona_mode == "🧠 Memory Mode":
            context = retrieve_context(user_input)
            context_text = "\n".join(context)
            mode_instruction = "Use memory context to answer accurately."

        elif persona_mode == "🤍 Emotional Support":
            context_text = ""
            mode_instruction = "Respond with emotional warmth and comfort."

        else:
            context_text = ""
            mode_instruction = "Respond casually and naturally."

        prompt = f"""
You are Jabez, a synthetic AI persona created for research.
Never claim to be a real human.
{mode_instruction}

Memory Context:
{context_text}

User: {user_input}
Jabez:
"""

        model_gemini = genai.GenerativeModel("gemini-1.5-flash")

        response = model_gemini.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 400,
            }
        )

        ai_text = response.text if response.text else "I'm here with you."

        emotion = detect_emotion(ai_text)

        st.session_state.chat.append(("ai", ai_text))

        # Save memory
        save_chat(user_input, ai_text)

        # Voice control by emotion
        if voice_on:
            slow_voice = True if emotion == "sad" else False
            audio_file = speak(ai_text, slow=slow_voice)
            st.audio(audio_file)

        if show_reasoning:
            st.markdown("### 🧠 Reasoning")
            st.write("Persona Mode:", persona_mode)
            st.write("Detected Emotion:", emotion)
