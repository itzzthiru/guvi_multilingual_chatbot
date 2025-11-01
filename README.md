# 🌍 GUVI Multilingual Chatbot

An AI-powered multilingual chatbot built with Streamlit and Hugging Face Transformers that can understand any major language, translate it to English, retrieve or generate the best possible answer, and respond back in the user’s own language — all in real time.  
---

## 🧠 Overview
This chatbot is designed to assist learners and visitors of the GUVI platform by providing instant, multilingual answers to their questions about GUVI courses, platforms like CodeKata, WebKata, SQLKata, certifications, and more.

It uses:

NLLB-200 for high-quality translation across 50+ languages.

Sentence-Transformers for FAQ and document semantic search.

GPT-powered fallback generation for open-ended questions.

A clean Streamlit UI for seamless interaction.
---

## 🧩 Features
Feature	Description

🌐 Multilingual Support	Detects and translates 50+ languages automatically.

🧠 Intelligent FAQ Retrieval	Searches GUVI FAQs using semantic embeddings.

📘 Content Search	Fetches the most relevant GUVI paragraph or document.

💬 GPT-Powered Fallback	Generates human-like answers if no FAQ or paragraph match is found.

🧾 Chat History	Maintains context during the session.

⚡ Fast & Lightweight	Uses MiniLM-L6-v2 for efficient sentence embeddings.

🖥️ Deployable Anywhere	Works seamlessly on Hugging Face Spaces or Streamlit Cloud.
---

## 🛠️ Tech Stack
Frontend: Streamlit

Backend Engines: Python, PyTorch

Models:

facebook/nllb-200-distilled-600M (Translation)

sentence-transformers/all-MiniLM-L6-v2 (Embeddings)

Hugging Face GPT/LLM Pipeline (Generative fallback)

Libraries: Transformers, Langdetect, Torch, Sentence-Transformers

Deployment: Hugging Face Spaces 
---

⚙️ Installation

# Clone the repository

git clone https://github.com/itzzthiru/guvi_multilingual_chatbot.git

cd guvi_multilingual_chatbot

# Install dependencies

pip install -r requirements.txt

# Run the app

streamlit run app.py
---

🚀 Deployment Links

Hugging Face Space: GUVI Multilingual Chatbot: https://huggingface.co/spaces/thiru43/guvi_chatbot

Demo Video: LinkedIn Demo: https://www.linkedin.com/posts/thirukumaran-undefined-8b336b352_project-title-guvi-multilingual-chatbot-activity-7361438830962323458-1hSl?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFf5eUcBIVM5jB52tTRYrgFrPIkAA2pI4tM

---

## 🧠 How It Works

User enters a question in any language.

Language is detected automatically (via langdetect).

Translated to English using NLLB-200.

FAQ and GUVI content are searched via sentence embeddings.

If no good match is found, GPT model generates a new answer.

Response is translated back to the user’s language.

Displayed neatly in Streamlit UI with confidence score.
---

## 👨‍💻 Author

Developed by Thirukumaran (Tk)

Data Science Student @ GUVI 
