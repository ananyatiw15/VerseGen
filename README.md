# VerseGen: Emotion-Aware Neural Poetry Generation

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Contribute!](#contribute)

## Overview

This project presents an emotion-aware poetry generation system that creates emotionally aligned poems based on user-defined prompts and emotions. Built using a custom-trained **hybrid RNN architecture**—comprising **Bidirectional LSTM**, **LSTM**, and an **Attention mechanism**—the model is trained on the **Gutenberg Poetry Corpus** and evaluated using sentiment-based validation. A user-friendly **Streamlit** interface allows real-time poem generation and emotion feedback.

## Features

- Generate poetry conditioned on both **topic** and **emotion**
- Emotion categories: **Joyful**, **Sad**, **Romantic**, **Fearful**
- Uses **VADER Sentiment Analyzer** to validate emotional alignment
- Emotion-tagged prompts with thematic seed starters
- Controlled creativity using a **temperature slider**
- Real-time interaction using a lightweight **Streamlit UI**

## Model Architecture

The neural network consists of the following components:

| Layer Type            | Description                               |
|-----------------------|-------------------------------------------|
| Embedding Layer       | 100-dimensional word embeddings           |
| Bidirectional LSTM    | Contextual learning from both directions  |
| LSTM Layer            | Sequential modeling of token sequences    |
| Attention Layer       | Focus mechanism for context representation|
| Dense Output Layer    | Vocabulary-sized softmax classification   |

Total Parameters: **15.28 million**  
Implementation Framework: **TensorFlow/Keras**

## Dataset

- **Source**: Gutenberg Poetry Corpus
- **Preprocessing**:
  - Tokenized and padded sequences
  - Emotion categories pseudo-labeled using sentiment scores
  - Maximum sequence length: 16 tokens

## How to Run

1. **Install dependencies**
   ```bash
   pip install streamlit==1.32.2
   pip install tensorflow==2.15.0
   pip install keras==2.15.0
   pip install numpy==1.24.3
   pip install nltk==3.8.1
   pip install scikit-learn==1.3.0
   
2. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   
3. **Usage**
- Enter a topic or poetic prompt
- Choose an emotion
- Adjust temperature for creativity
- Click "Generate Poem" to view results and emotion score

## Contribute!

Contributions are welcome! If you'd like to enhance the model, add new emotion categories, or improve the UI, feel free to fork the repository and submit a pull request. Whether it's fixing bugs, adding features, or sharing feedback—every bit helps make this project better.


Feel free to reach out:
 
- **LinkedIn**: [Ananya Tiwari](https://linkedin.com/in/ananya-tiw)  

Let's build something poetic and powerful together!
