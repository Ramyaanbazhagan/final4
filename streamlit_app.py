import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Jabez – Neural Persona", layout="wide")

# =====================================
# 🔐 PUT YOUR API KEY HERE
# =====================================
genai.configure(api_key="AIzaSyAfxdwvR3OA6Cuki9b3JOyHmsNeFIkyLGs")

# =====================================
# LOAD DATASET
# =====================================
@st.cache_resource
def load_memory():
    with open("dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

memory_data = load_memory()

# =====================================
# FLATTEN MEMORY SAFELY
# =====================================
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
            text = str(obj)
            if len(text) < 500:  # prevent huge chunks
                texts.append(text)

    extract(data)
    return texts

memory_texts = flatten_memory(memory_data)

# =====================================
# LOAD EMBEDDINGS
# =====================================
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(memory_texts, convert_to_tensor=True)
    return model, embeddings

embed_model, embeddings = load_embeddings()

# =====================================
# MEMORY RETRIEVAL
# =====================================
def retrieve_context(query, top_k=2):
    if not memory_texts:
        return []

    query_emb = embed_model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:top_k]
    return [memory_texts[i][:300] for i in top_idx]

# =====================================
# EMOTION DETECTION
# =====================================
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["sad", "lonely", "miss", "cry", "upset"]):
        return "sad"
    if any(w in t for w in ["happy", "love", "excited", "great"]):
        return "happy"
    return "neutral"

# =====================================
# TEXT TO SPEECH
# =====================================
def speak(text):
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("🧠 Jabez Control Panel")

mode = st.sidebar.radio(
    "Persona Mode",
    ["🧠 Memory Mode", "💬 Casual Talk", "🤍 Emotional Support"]
)

voice_on = st.sidebar.checkbox("🔊 Voice Output", value=True)
show_reasoning = st.sidebar.checkbox("🧠 Show Reasoning")

st.sidebar.markdown("---")
st.sidebar.caption("Ethical AI • Responsible AI • Academic Prototype")

# =====================================
# MAIN UI
# =====================================
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

# =====================================
# USER INPUT
# =====================================
st.markdown("---")
user_input = st.text_input("Talk to Jabez:")

if st.button("Send"):
    if user_input.strip():

        st.session_state.chat.append(("user", user_input))

        # Emotion
        emotion = detect_emotion(user_input)

        # Retrieve limited memory
        context = retrieve_context(user_input)
        context_text = "\n".join(context)

        # Tone control
        if emotion == "sad":
            tone = "Respond softly and briefly in a comforting way."
        elif emotion == "happy":
            tone = "Respond energetically and warmly."
        else:
            tone = "Respond naturally and warmly."

        # Mode control
        if mode == "🧠 Memory Mode":
            mode_instruction = "Use the memory context carefully."
        elif mode == "🤍 Emotional Support":
            mode_instruction = "Focus on emotional reassurance."
        else:
            mode_instruction = "Have a light casual conversation."

        # SAFE SHORT PROMPT
        prompt = f"""
You are Jabez, a synthetic AI persona.
Do not claim to be human.
Stay ethical and avoid emotional dependency.

{tone}
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
        except Exception as e:
            ai_text = f"API Error: {str(e)}"

        st.session_state.chat.append(("ai", ai_text))

        if voice_on and "API Error" not in ai_text:
            audio_file = speak(ai_text)
            st.audio(audio_file)

        if show_reasoning:
            st.markdown("### 🧠 Debug Info")
            st.write("Emotion:", emotion)
            st.write("Mode:", mode)
            st.write("Memory Used:", context)
