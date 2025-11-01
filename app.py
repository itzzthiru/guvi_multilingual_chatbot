# app.py
import streamlit as st
from chatbot import Chatbot
import time

st.set_page_config(page_title="GUVI Multilingual Chatbot", page_icon="🤖", layout="centered")

# Use a cached resource so heavy models initialize only once.
@st.cache_resource(show_spinner=False)
def get_bot():
    return Chatbot()

def init_session():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (speaker, text)
    if "last_lang" not in st.session_state:
        st.session_state.last_lang = "eng_Latn"

def render_chat():
    for speaker, text in st.session_state.history:
        if speaker == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")

def main():
    init_session()
    st.title("🤖 GUVI Multilingual Chatbot — GUVI Project")
    st.caption("Type in English or your native language. I translate, search GUVI content/FAQ, and reply in your language.")
    st.markdown("---")

    # Controls: language hint and options
    cols = st.columns([3, 1])
    with cols[0]:
        user_input = st.chat_input("Type your message here...")  # available in newer Streamlit versions
    with cols[1]:
        # show detected language from last response if available
        st.markdown(f"**Detected:** {st.session_state.get('last_lang', 'N/A')}")

    bot = get_bot()
    if user_input:
        user_input = user_input.strip()
        st.session_state.history.append(("user", user_input))
        # call bot
        with st.spinner("Processing..."):
            try:
                start = time.time()
                response_data = bot.get_response(user_input, top_k=3)
                latency = time.time() - start

                # store detected language for UI hint
                st.session_state.last_lang = response_data.get("detected_lang_code", st.session_state.last_lang)

                # collect combined outputs (FAQ top-k, GUVI top-k, generative fallback)
                outputs = []

                # FAQ answers (may contain list of tuples (answer, score))
                faqs = response_data.get("faq_answers", [])
                if faqs:
                    outputs.append(("FAQ", faqs))

                # GUVI paragraphs
                guvi = response_data.get("guvi_paragraphs", [])
                if guvi:
                    outputs.append(("GUVI", guvi))

                # Generative fallback might be present
                gen = response_data.get("generative_answers", [])
                if gen:
                    outputs.append(("Generative", gen))

                # Format display: show top-k grouped and confidence scores
                display_texts = []
                for kind, items in outputs:
                    st.markdown(f"**{kind} results:**")
                    for idx, item in enumerate(items):
                        # item could be (text, score, optional_source)
                        if isinstance(item, (list, tuple)):
                            text = item[0]
                            score = item[1] if len(item) > 1 else None
                            tag = item[2] if len(item) > 2 else ""
                        else:
                            text = str(item)
                            score = None
                            tag = ""
                        s = f"- {text}"
                        if score is not None:
                            s += f"  _(score: {score:.3f})_"
                        if tag:
                            s += f"  _[{tag}]_"
                        st.markdown(s)

                if not outputs:
                    st.markdown("I couldn't find a close match. I tried FAQ and GUVI documents. Try rephrasing.")

                st.caption(f"Latency: {latency:.2f}s")
                # Save the last Bot output in history: choose primary result or generative fallback
                # For simplicity, if generative exists, show that as primary; else show first available
                primary_text = None
                if gen:
                    primary_text = gen[0][0]
                elif faqs:
                    primary_text = faqs[0][0]
                elif guvi:
                    primary_text = guvi[0][0]

                if primary_text:
                    st.session_state.history.append(("bot", primary_text))
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.history.append(("bot", "Sorry, something went wrong."))

    # Render chat history below input area
    st.markdown("---")
    st.header("Conversation")
    render_chat()

if __name__ == "__main__":
    main()
