# VerseGen: Emotion-Aware Neural Poetry Generation

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [How to Run](#how-to-run)

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
   pip install -r requirements.txt
