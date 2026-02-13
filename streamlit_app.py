import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Jabez – Neural Persona", layout="wide")

# ===============================
# 🔐 API KEY (PUT YOUR KEY HERE)
# ===============================
genai.configure(api_key="AIzaSyDSALqpEtbhaBXDfmPFvgxwBI7xmagDZow")

# ===============================
# LOAD DATASET
# ===============================
@st.cache_resource
def load_memory():
    with open("dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

memory_data = load_memory()

# ===============================
# FLATTEN MEMORY
# ===============================
def flatten_memory(data):
    texts = []

    def extract(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                extract(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item)
        else:
            texts.append(str(obj))

    extract(data)
    return texts

memory_texts = flatten_memory(memory_data)

# ===============================
# LOAD EMBEDDINGS
# ===============================
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(memory_texts, convert_to_tensor=True)
    return model, embeddings

embed_model, embeddings = load_embeddings()

# ===============================
# MEMORY RETRIEVAL
# ===============================
def retrieve_context(query, top_k=3):
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:top_k]
    return [memory_texts[i] for i in top_idx]

# ===============================
# EMOTION DETECTION
# ===============================
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["sad", "miss", "lonely", "cry", "upset"]):
        return "sad"
    if any(w in t for w in ["happy", "excited", "love", "great"]):
        return "happy"
    return "neutral"

# ===============================
# TEXT TO SPEECH
# ===============================
def speak(text):
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# ===============================
# SIDEBAR CONTROL PANEL
# ===============================
st.sidebar.title("🧠 Jabez Control Panel")

mode = st.sidebar.radio(
    "Persona Mode",
    ["🧠 Memory Mode", "💬 Casual Talk", "🤍 Emotional Support"]
)

voice_on = st.sidebar.checkbox("🔊 Voice Output", value=True)
show_reasoning = st.sidebar.checkbox("🧠 Show Reasoning")

st.sidebar.markdown("---")
st.sidebar.caption("Ethical AI • Responsible AI • Research Prototype")

# ===============================
# MAIN UI
# ===============================
st.title("🤖 Jabez – Neural Persona")

col1, col2 = st.columns([1, 2])

with col1:
    st.image(
        "https://i.imgur.com/8Km9tLL.png",
        caption="Jabez Persona",
        use_column_width=True
    )

with col2:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, msg in st.session_state.chat[-6:]:
        if role == "user":
            st.markdown(f"🧍 **You:** {msg}")
        else:
            st.markdown(f"🤖 **Jabez:** {msg}")

# ===============================
# USER INPUT
# ===============================
st.markdown("---")
user_input = st.text_input("Talk to Jabez:")

if st.button("Send"):
    if user_input.strip():

        st.session_state.chat.append(("user", user_input))

        # Retrieve limited memory
        context = retrieve_context(user_input, top_k=3)
        context_text = "\n".join(context[:3])

        emotion = detect_emotion(user_input)

        # Tone control
        if emotion == "sad":
            tone_instruction = "Respond softly, gently, briefly, and comfortingly."
        elif emotion == "happy":
            tone_instruction = "Respond energetically and warmly."
        else:
            tone_instruction = "Respond naturally and warmly."

        # Mode control
        if mode == "🧠 Memory Mode":
            mode_instruction = "Use memory context carefully."
        elif mode == "🤍 Emotional Support":
            mode_instruction = "Focus on emotional reassurance."
        else:
            mode_instruction = "Have a light and casual conversation."

        prompt = f"""
You are Jabez, a synthetic AI persona created for research.

IMPORTANT:
- Do NOT claim to be human.
- Do NOT create emotional dependency.
- Stay within ethical AI boundaries.

{tone_instruction}
{mode_instruction}

Memory Context:
{context_text}

User: {user_input}
Jabez:
"""

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")

            response = model.generate_content(
                prompt[:8000]  # limit size to avoid error
            )

            ai_text = response.text.strip()

        except Exception as e:
            ai_text = "Sorry, I encountered an issue generating a response."

        st.session_state.chat.append(("ai", ai_text))

        # Voice
        if voice_on:
            audio_file = speak(ai_text)
            st.audio(audio_file)

        # Reasoning
        if show_reasoning:
            st.markdown("### 🧠 Internal State")
            st.write("Emotion detected:", emotion)
            st.write("Mode:", mode)
            st.write("Memory Used:")
            for c in context:
                st.write("-", c)
