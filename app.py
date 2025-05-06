import streamlit as st
import numpy as np
import pickle
import re
import random
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

# Set Streamlit page config
st.set_page_config(page_title="Poetic Emotions 🌙", layout="centered")

# ─────────────────────────────────────────────
# Load Model, Tokenizer, VADER
# ─────────────────────────────────────────────
@st.cache_resource
def load_resources():
    class Attention(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(Attention, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                     initializer='random_normal', trainable=True)
            self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                     initializer='zeros', trainable=True)
            super().build(input_shape)

        def call(self, inputs):
            e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
            e = tf.keras.backend.squeeze(e, axis=-1)
            alpha = tf.keras.backend.softmax(e)
            alpha = tf.keras.backend.expand_dims(alpha, axis=-1)
            context = inputs * alpha
            context = tf.keras.backend.sum(context, axis=1)
            return context

    model = load_model("poem_generator_best.keras", custom_objects={"Attention": Attention})
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    analyzer = SentimentIntensityAnalyzer()
    return model, tokenizer, analyzer

model, tokenizer, sentiment_analyzer = load_resources()

# ─────────────────────────────────────────────
# Setup emotion tags & blacklist
# ─────────────────────────────────────────────
emotion_tokens = {
    "joyful": "<joyful>",
    "sad": "<sad>",
    "romantic": "<romantic>",
    "fearful": "<fearful>"
}

emotion_score_ranges = {
    "joyful": (0.3, 1.0),
    "sad": (-1.0, -0.2),
    "romantic": (0.1, 1.0),
    "fearful": (-1.0, 0.0)
}

blacklist_words = {"phiz", "cam", "rum", "thy", "thou", "hast", "slain", "ere"}

# ─────────────────────────────────────────────
# Clean prompt and enrich with emotion-based starter
# ─────────────────────────────────────────────
def prompt_cleaner(user_prompt, emotion="joyful"):
    user_prompt = user_prompt.lower()
    user_prompt = re.sub(r"[^a-zA-Z0-9\s]", "", user_prompt)

    patterns = [
        r"write a poem about (.+)",
        r"poem about (.+)",
        r"generate a poem on (.+)",
        r"poem on (.+)",
        r"romantic poem about (.+)",
        r".*about (.+)",
    ]

    topic = None
    for pattern in patterns:
        match = re.search(pattern, user_prompt)
        if match:
            topic = match.group(1).strip()
            break

    if not topic:
        return user_prompt, user_prompt

    emotion_starters = {
        "joyful": [
            f"{topic} dances in golden light",
            f"{topic} blooms in endless spring",
            f"{topic} sings like morning birds"
        ],
        "sad": [
            f"{topic} weeps beneath the gray sky",
            f"{topic} fades like autumn leaves",
            f"{topic} sleeps in shadows cold"
        ],
        "romantic": [
            f"{topic} whispers in candlelight dreams",
            f"{topic} glows in lovers’ gaze",
            f"{topic} dances with soft moonlight"
        ],
        "fearful": [
            f"{topic} creeps through haunted fog",
            f"{topic} hides in shadows deep",
            f"{topic} trembles beneath the storm"
        ]
    }

    starter = random.choice(emotion_starters.get(emotion, []))
    return starter, topic

# ─────────────────────────────────────────────
# Format poem
# ─────────────────────────────────────────────
def format_poem(poem, words_per_line=7):
    words = poem.split()
    lines = [' '.join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    return '\n'.join(line.capitalize() for line in lines)

# ─────────────────────────────────────────────
# Analyze emotion using VADER
# ─────────────────────────────────────────────
def analyze_emotion(poem, analyzer):
    scores = analyzer.polarity_scores(poem)
    return scores["compound"]

def emotion_matches(compound_score, target_emotion):
    lower, upper = emotion_score_ranges.get(target_emotion, (0.0, 1.0))
    return lower <= compound_score <= upper

def contains_blacklisted_words(poem):
    return any(word in poem.lower().split() for word in blacklist_words)

# ─────────────────────────────────────────────
# Generate poem with retry loop and blacklist
# ─────────────────────────────────────────────
def generate_poem_with_checks(prompt, emotion, model, tokenizer, analyzer, temperature=0.75, max_retries=5):
    best_poem = ""
    best_score = -1.0

    for attempt in range(max_retries):
        poem = generate_poem(prompt, emotion, model, tokenizer, temperature)
        compound_score = analyze_emotion(poem, analyzer)

        if contains_blacklisted_words(poem):
            continue  # Skip weird outputs

        if emotion_matches(compound_score, emotion):
            return poem, compound_score

        if compound_score > best_score:
            best_score = compound_score
            best_poem = poem

    return best_poem, best_score  # Fallback if no exact match

# ─────────────────────────────────────────────
# Core poem generation logic
# ─────────────────────────────────────────────
def generate_poem(prompt, emotion, model, tokenizer, temperature=0.75, next_words=50):
    seed_text, topic = prompt_cleaner(prompt, emotion)
    emotion_tag = emotion_tokens.get(emotion.lower(), "<joyful>")
    seed_text = f"{emotion_tag} {seed_text.lower()} {topic} {topic}"
    max_seq_len = model.input_shape[1]

    recent_bigrams = set()
    recent_trigrams = set()

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len, padding='pre')

        predictions = model.predict(token_list, verbose=0)[0]
        predictions = np.log(predictions + 1e-8) / temperature
        exp_preds = np.exp(predictions)
        probabilities = exp_preds / np.sum(exp_preds)

        sorted_indices = np.argsort(probabilities)[::-1]

        for idx in sorted_indices:
            next_word = tokenizer.index_word.get(idx, "")
            if not next_word:
                continue

            current_tokens = seed_text.split()
            bigram = tuple(current_tokens[-1:] + [next_word])
            trigram = tuple(current_tokens[-2:] + [next_word]) if len(current_tokens) >= 2 else ()

            if bigram in recent_bigrams or trigram in recent_trigrams:
                continue

            seed_text += " " + next_word
            recent_bigrams.add(bigram)
            if trigram:
                recent_trigrams.add(trigram)
            break

    poem = seed_text.replace(emotion_tag, "").strip()
    return format_poem(poem)

# ─────────────────────────────────────────────
# 🌐 Streamlit UI
# ─────────────────────────────────────────────
st.title("🎭 Emotion-Aware Poetry Generator")
st.markdown("Craft poetic verses guided by your theme and emotion.\nPowered by a custom-trained LSTM + Attention model.")

user_input = st.text_input("✍️ Enter a poetic topic or phrase:")
selected_emotion = st.selectbox("🎨 Choose an emotion:", ["joyful", "sad", "romantic", "fearful"])
temperature = st.slider("🔥 Creativity (Temperature):", min_value=0.4, max_value=1.2, value=0.7, step=0.05)

if st.button("🪄 Generate Poem"):
    if not user_input.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Weaving your poetic story..."):
            poem, score = generate_poem_with_checks(user_input, selected_emotion, model, tokenizer, sentiment_analyzer, temperature)
        st.markdown("### 📜 Your Poem:")
        st.text_area("Poem Output", value=poem, height=300)
        st.markdown(f"### 🧠 Emotion Score: `{score:.2f}` ({selected_emotion.title()} match)")
